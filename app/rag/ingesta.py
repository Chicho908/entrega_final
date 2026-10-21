# Módulo de ingesta: lee los PDF, genera los embeddings con Pinecone Inference y sube los vectores al índice de Pinecone
# Se ejecuta UNA vez antes de usar el sistema: python -m app.rag.ingesta   (docker compose lo corre solo, como paso previo a la API y al worker)
# Es idempotente: los ids de los fragmentos salen del contenido, así que volver a correrlo sobrescribe en vez de duplicar

# Importa asyncio para correr la ingesta asíncrona desde el punto de entrada
import asyncio
# Importa logging para dejar registro de cada paso
import logging
# Importa sys para devolver un código de salida distinto de cero si falla (docker compose lo usa para no levantar la API)
import sys

# Importa el cliente asíncrono oficial de Pinecone
from pinecone import PineconeAsyncio

# Importa la configuración y su error propio
from app.config import ConfigError, Settings, cargar_configuracion
# Importa el logging común de la API y el worker
from app.observability import configurar_logging
# Importa las piezas del RAG
from app.rag.almacen import AlmacenPinecone, ErrorIndice, asegurar_indice
from app.rag.corpus import cargar_chunks_async
from app.rag.embeddings import EmbedderPinecone

# Crea el logger del módulo
logger = logging.getLogger(__name__)


# Ingesta completa: devuelve la cantidad de fragmentos subidos
async def ingestar(settings: Settings) -> int:
    # 1) Lee y trocea los PDF (en un hilo: es trabajo de CPU y disco)
    chunks = await cargar_chunks_async(settings.datos_dir, settings.chunk_size, settings.chunk_overlap)
    logger.info("1/4 Corpus leído: %s fragmentos", len(chunks))
    async with PineconeAsyncio(api_key=settings.pinecone_api_key) as cliente:
        # 2) Crea el índice si no existe y comprueba que la dimensión coincida con la configuración
        host = await asegurar_indice(cliente, settings.pinecone_index, settings.embedding_dim, settings.pinecone_cloud, settings.pinecone_region)
        logger.info("2/4 Índice '%s' listo", settings.pinecone_index)
        # 3) Genera los embeddings con el modelo configurado
        vectores = await EmbedderPinecone(cliente, settings.embedding_model).embeber_documentos([chunk.texto for chunk in chunks])
        # Segunda barrera contra el error de "mismatch de dimensiones": el modelo real tiene que dar la dimensión configurada
        dimensiones = {len(vector) for vector in vectores}
        if dimensiones != {settings.embedding_dim}:
            raise ErrorIndice(f"El modelo '{settings.embedding_model}' devolvió vectores de dimensión {sorted(dimensiones)}, pero EMBEDDING_DIM vale {settings.embedding_dim}.")
        logger.info("3/4 Embeddings generados con '%s' (dimensión %s)", settings.embedding_model, settings.embedding_dim)
        # 4) Sube los vectores con su metadata al namespace configurado
        async with cliente.IndexAsyncio(host=host) as indice:
            subidos = await AlmacenPinecone(indice, settings.pinecone_namespace).guardar(chunks, vectores)
    logger.info("4/4 Ingesta terminada: %s vectores en '%s' / '%s'", subidos, settings.pinecone_index, settings.pinecone_namespace)
    return subidos


# Punto de entrada: python -m app.rag.ingesta
def main() -> int:
    configurar_logging()
    try:
        settings = cargar_configuracion()
        asyncio.run(ingestar(settings))
    except ConfigError as error:
        logger.error("Error de configuración: %s", error)
        return 1
    except Exception:
        # Cualquier otra falla (red, API key de Pinecone inválida, dimensión incorrecta) corta con un código distinto de cero
        logger.exception("La ingesta falló")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
