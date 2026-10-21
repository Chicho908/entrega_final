# Módulo de runtime: arma los recursos compartidos de un proceso (conexión a Redis, almacén de trabajos y grafo con su checkpointer)
# Lo usan igual la API (si levanta workers propios) y el proceso worker, así los dos hablan con el mismo Redis

# Importa AsyncIterator para tipar el context manager
from collections.abc import AsyncIterator
# Importa AsyncExitStack y asynccontextmanager para abrir y cerrar los recursos ordenadamente
from contextlib import AsyncExitStack, asynccontextmanager
# Importa dataclass para agrupar los recursos
from dataclasses import dataclass

# Importa el cliente asíncrono oficial de Pinecone
from pinecone import PineconeAsyncio
# Importa el checkpointer asíncrono de Redis (guarda el estado del grafo en cada paso; requiere Redis Stack: módulos RediSearch y RedisJSON)
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
# Importa el cliente asíncrono de Redis
from redis.asyncio import Redis

# Importa la configuración
from app.config import Settings
# Importa el grafo y los LLM
from app.graph import GrafoCompilado, LLMs, compilar_orquestador
# Importa el despachador de la API
from app.hitl import DespachadorRedis
# Importa el almacén de trabajos
from app.jobs import JobStore
# Importa la fábrica del LLM
from app.llm import crear_llm
# Importa las piezas del RAG
from app.rag.almacen import AlmacenPinecone, obtener_host
from app.rag.corpus import cargar_chunks_async
from app.rag.embeddings import EmbedderPinecone
from app.rag.recuperador import RecuperadorHibrido


# Recursos de un proceso
@dataclass(frozen=True)
class Runtime:
    # Configuración validada
    settings: Settings
    # Almacén de trabajos y cola (Redis)
    store: JobStore
    # Grafo compilado con checkpointer; None en una API que no ejecuta trabajos (los ejecutan los workers)
    grafo: GrafoCompilado | None


# Abre los recursos y los cierra al salir. con_grafo=False sirve para una API "liviana" que solo encola y consulta
@asynccontextmanager
async def abrir_runtime(settings: Settings, con_grafo: bool) -> AsyncIterator[Runtime]:
    # Cliente de Redis con respuestas en texto
    redis: Redis = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        # Falla rápido y claro si Redis no está disponible, en vez de fallar en la primera petición
        await redis.ping()
        store = JobStore(redis, settings.job_ttl_s)
        # API liviana: no necesita LLM ni checkpointer
        if not con_grafo:
            yield Runtime(settings, store, None)
            return
        # Un mismo LLM para todos los roles (se puede separar por rol, ver PE6)
        llm = crear_llm(settings)
        llms = LLMs(supervisor=llm, investigador=llm, analista=llm, sintetizador=llm)
        # El TTL del checkpointer está en minutos
        ttl = {"default_ttl": max(1, settings.job_ttl_s // 60)}
        # La pila cierra en orden inverso todo lo que se abre: checkpointer, índice y cliente de Pinecone
        async with AsyncExitStack() as pila:
            # Cliente asíncrono de Pinecone + índice (debe existir: lo crea la ingesta). Falla rápido si la key o el índice son inválidos
            pinecone = await pila.enter_async_context(PineconeAsyncio(api_key=settings.pinecone_api_key))
            host = await obtener_host(pinecone, settings.pinecone_index, settings.embedding_dim)
            indice = await pila.enter_async_context(pinecone.IndexAsyncio(host=host))
            # Corpus local (para BM25 y para devolver el texto de cada fragmento) y recuperador híbrido
            chunks = await cargar_chunks_async(settings.datos_dir, settings.chunk_size, settings.chunk_overlap)
            recuperador = RecuperadorHibrido(
                chunks,
                EmbedderPinecone(pinecone, settings.embedding_model),
                AlmacenPinecone(indice, settings.pinecone_namespace),
                rrf_k=settings.rrf_k,
            )
            checkpointer = await pila.enter_async_context(AsyncRedisSaver.from_conn_string(settings.redis_url, ttl=ttl))
            grafo = compilar_orquestador(llms, settings, checkpointer, recuperador, DespachadorRedis(redis))
            yield Runtime(settings, store, grafo)
    finally:
        await redis.aclose()
