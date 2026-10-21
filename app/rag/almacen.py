# Módulo del almacén vectorial: guarda y consulta los vectores en un índice serverless de Pinecone con el SDK asíncrono oficial
# El nombre del índice, el namespace, la nube/región y la dimensión salen de la configuración (variables de entorno), nunca del código

# Importa logging para dejar registro de lo que se crea y se sube
import logging
# Importa Sequence para tipar las listas
from collections.abc import Sequence
# Importa Any y Protocol para tipar el contrato y el índice del SDK
from typing import Any, Protocol

# Importa el cliente asíncrono de Pinecone y la especificación del índice serverless
from pinecone import PineconeAsyncio, ServerlessSpec

# Importa el fragmento del corpus
from app.rag.corpus import Chunk
# Importa los reintentos con log
from app.rag.reintentos import con_reintentos

# Crea el logger del módulo
logger = logging.getLogger(__name__)

# Métrica de similitud del índice (coseno es la que recomienda Pinecone para los modelos e5)
METRICA = "cosine"
# Cantidad de vectores por pedido de upsert
LOTE_UPSERT = 100


# Error propio: el índice existe pero no coincide con la configuración (por ejemplo, otra dimensión)
class ErrorIndice(Exception):
    pass


# Contrato de un almacén vectorial (los tests usan uno en memoria; en producción es Pinecone)
class AlmacenVectorial(Protocol):
    # Guarda los fragmentos con sus vectores y devuelve cuántos guardó
    async def guardar(self, chunks: Sequence[Chunk], vectores: Sequence[Sequence[float]]) -> int: ...

    # Busca los k vectores más parecidos (opcionalmente solo de una categoría) y devuelve (chunk_id, similitud)
    async def buscar(self, vector: Sequence[float], k: int, categoria: str | None = None) -> list[tuple[str, float]]: ...


# Se asegura de que el índice exista con la dimensión correcta y devuelve su host (lo crea si hace falta)
async def asegurar_indice(cliente: PineconeAsyncio, nombre: str, dimension: int, cloud: str, region: str) -> str:
    # Crea el índice solo si no existe (la creación espera hasta que está listo)
    if not await con_reintentos(lambda: cliente.has_index(nombre), "consultar si existe el índice"):
        logger.info("Creando el índice '%s' (dimensión %s, %s/%s)...", nombre, dimension, cloud, region)
        await cliente.create_index(name=nombre, dimension=dimension, metric=METRICA, spec=ServerlessSpec(cloud=cloud, region=region))
    descripcion = await con_reintentos(lambda: cliente.describe_index(nombre), "describir el índice")
    # Error clásico de los RAG: embeddings de una dimensión contra un índice de otra. Se detecta acá, antes de subir nada
    if descripcion.dimension != dimension:
        raise ErrorIndice(
            f"El índice '{nombre}' tiene dimensión {descripcion.dimension}, pero EMBEDDING_DIM vale {dimension}. "
            "Usá otro PINECONE_INDEX_NAME o corregí EMBEDDING_DIM para que coincidan con el modelo de embeddings."
        )
    return str(descripcion.host)


# Devuelve el host de un índice que YA existe (la API y el worker no crean índices: eso lo hace la ingesta)
async def obtener_host(cliente: PineconeAsyncio, nombre: str, dimension: int) -> str:
    if not await con_reintentos(lambda: cliente.has_index(nombre), "consultar si existe el índice"):
        raise ErrorIndice(f"El índice '{nombre}' no existe en Pinecone. Corré primero la ingesta: python -m app.rag.ingesta (docker compose la corre sola).")
    descripcion = await con_reintentos(lambda: cliente.describe_index(nombre), "describir el índice")
    if descripcion.dimension != dimension:
        raise ErrorIndice(f"El índice '{nombre}' tiene dimensión {descripcion.dimension}, pero EMBEDDING_DIM vale {dimension}.")
    return str(descripcion.host)


# Almacén real: un namespace de un índice de Pinecone
class AlmacenPinecone:
    # Guarda el índice ya abierto (cliente de datos asíncrono) y el namespace
    def __init__(self, indice: Any, namespace: str) -> None:
        self.indice = indice
        self.namespace = namespace

    # Sube los fragmentos con su metadata (fuente, página, categoría, sección). Es idempotente: el mismo id se sobreescribe
    async def guardar(self, chunks: Sequence[Chunk], vectores: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(vectores):
            raise ValueError(f"Hay {len(chunks)} fragmentos pero {len(vectores)} vectores.")
        registros = [
            {
                "id": chunk.chunk_id,
                "values": list(vector),
                "metadata": {"fuente": chunk.fuente, "pagina": chunk.pagina, "categoria": chunk.categoria, "seccion": chunk.seccion},
            }
            for chunk, vector in zip(chunks, vectores, strict=True)
        ]
        for inicio in range(0, len(registros), LOTE_UPSERT):
            lote = registros[inicio : inicio + LOTE_UPSERT]

            # Función sin argumentos para poder reintentarla
            async def subir(lote: list[dict[str, Any]] = lote) -> Any:
                return await self.indice.upsert(vectors=lote, namespace=self.namespace)

            await con_reintentos(subir, "upsert en Pinecone")
        logger.info("%s vectores subidos al namespace '%s'", len(registros), self.namespace)
        return len(registros)

    # Busca por similitud; el filtro por metadata (categoría) lo resuelve Pinecone antes de ordenar
    async def buscar(self, vector: Sequence[float], k: int, categoria: str | None = None) -> list[tuple[str, float]]:
        filtro = {"categoria": {"$eq": categoria}} if categoria else None

        # Función sin argumentos para poder reintentarla
        async def consultar() -> Any:
            return await self.indice.query(vector=list(vector), top_k=k, namespace=self.namespace, filter=filtro, include_metadata=False)

        respuesta = await con_reintentos(consultar, "query en Pinecone")
        return [(str(coincidencia["id"]), float(coincidencia["score"])) for coincidencia in respuesta["matches"]]
