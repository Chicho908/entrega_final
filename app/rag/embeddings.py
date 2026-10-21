# Módulo de embeddings: convierte texto en vectores con Pinecone Inference (el modelo se elige por variable de entorno)
# Todo es asíncrono: se usa el SDK oficial asíncrono de Pinecone, así no hay llamadas bloqueantes dentro de funciones async

# Importa Sequence para tipar las listas de textos
from collections.abc import Sequence
# Importa Any y Protocol para tipar el contrato y la respuesta del SDK
from typing import Any, Literal, Protocol

# Importa el cliente asíncrono oficial de Pinecone
from pinecone import PineconeAsyncio

# Importa los reintentos con log
from app.rag.reintentos import con_reintentos

# Cantidad máxima de textos por pedido de embeddings (el límite de Pinecone Inference para modelos de texto es 96)
LOTE_MAXIMO = 96


# Contrato de un generador de embeddings (los tests usan uno falso; en producción es Pinecone Inference)
class Embedder(Protocol):
    # Vectores de varios textos del corpus (los "pasajes")
    async def embeber_documentos(self, textos: Sequence[str]) -> list[list[float]]: ...

    # Vector de una consulta del usuario
    async def embeber_consulta(self, texto: str) -> list[float]: ...


# Extrae los valores numéricos de un embedding del SDK (el objeto se comporta como diccionario y como objeto)
def _valores(embedding: Any) -> list[float]:
    return [float(x) for x in embedding["values"]]


# Embedder real: usa pc.inference.embed de Pinecone
class EmbedderPinecone:
    # Guarda el cliente de Pinecone (uno por proceso, ya abierto) y el modelo de embeddings
    def __init__(self, cliente: PineconeAsyncio, modelo: str) -> None:
        self.cliente = cliente
        self.modelo = modelo

    # Pide los embeddings de un lote de textos; input_type distingue pasajes de consultas (los modelos e5 lo necesitan)
    async def _embeber(self, textos: Sequence[str], tipo: Literal["passage", "query"]) -> list[list[float]]:
        vectores: list[list[float]] = []
        for inicio in range(0, len(textos), LOTE_MAXIMO):
            lote = list(textos[inicio : inicio + LOTE_MAXIMO])

            # Función sin argumentos que reintenta con_reintentos si la red falla
            async def pedir(lote: list[str] = lote) -> Any:
                return await self.cliente.inference.embed(model=self.modelo, inputs=lote, parameters={"input_type": tipo, "truncate": "END"})

            respuesta = await con_reintentos(pedir, f"embeddings de Pinecone ({tipo})")
            vectores.extend(_valores(embedding) for embedding in respuesta)
        return vectores

    # Embeddings de los fragmentos del corpus
    async def embeber_documentos(self, textos: Sequence[str]) -> list[list[float]]:
        return await self._embeber(textos, "passage")

    # Embedding de la pregunta del usuario
    async def embeber_consulta(self, texto: str) -> list[float]:
        return (await self._embeber([texto], "query"))[0]
