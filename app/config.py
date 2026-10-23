# Módulo de configuración: lee las variables de entorno y las valida apenas arranca el programa
# Nada de lo que cambia entre entornos (modelos, índice de Pinecone, namespace, claves) está escrito en el código: todo sale de acá

# Importa os para leer las variables de entorno
import os
# Importa dataclass y field para agrupar toda la configuración en un solo objeto inmutable
from dataclasses import dataclass, field
# Importa Literal, cast y get_args para restringir y validar los proveedores de LLM permitidos
from typing import Literal, cast, get_args

# Importa load_dotenv para cargar el archivo .env cuando se trabaja en local
from dotenv import load_dotenv

# Define los proveedores de LLM que soporta el proyecto
Proveedor = Literal["openai", "anthropic"]

# Define el nombre de la variable de entorno donde vive la API key de cada proveedor
VARIABLE_API_KEY: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


# Error propio para avisar de forma clara que una variable de entorno falta o es inválida
class ConfigError(Exception):
    pass


# Agrupa toda la configuración del proyecto en un objeto inmutable (frozen=True)
@dataclass(frozen=True)
class Settings:
    # Proveedor de LLM elegido (openai o anthropic)
    proveedor: Proveedor
    # API key del proveedor (repr=False evita que se imprima por accidente en logs o errores)
    api_key: str = field(repr=False)
    # Nombre del modelo de chat a usar
    modelo: str
    # Tope de pasos del grafo principal por invocación (red de seguridad contra bucles infinitos)
    limite_recursion: int
    # Tope de decisiones del supervisor: al superarlo se fuerza la síntesis con lo que haya
    max_pasos_supervisor: int
    # Veces que un especialista puede repetir su trabajo para corregir un error de validación
    max_refinamientos: int
    # Tope de pasos del ciclo ReAct interno de cada especialista
    limite_iteraciones_agente: int
    # API key de Pinecone (índice vectorial y embeddings; repr=False para que no se imprima)
    pinecone_api_key: str = field(default="", repr=False)
    # Nombre del índice de Pinecone donde viven los vectores de las políticas
    pinecone_index: str = ""
    # Namespace dentro del índice (separa este corpus de cualquier otro)
    pinecone_namespace: str = ""
    # Nube y región del índice serverless (solo se usan si el índice hay que crearlo; siempre salen de PINECONE_CLOUD / PINECONE_REGION, sin valores escritos en el código)
    pinecone_cloud: str = ""
    pinecone_region: str = ""
    # Modelo de embeddings de Pinecone Inference y su dimensión (el índice se crea con esta dimensión)
    embedding_model: str = ""
    embedding_dim: int = 0
    # Carpeta con los PDF de las políticas
    datos_dir: str = "data"
    # Cantidad de fragmentos que devuelve el recuperador híbrido
    top_k: int = 4
    # Constante k de la fusión RRF (60 es el valor estándar de la literatura)
    rrf_k: int = 60
    # Tamaño máximo (en caracteres) de un fragmento y solapamiento cuando una sección hay que partirla
    chunk_size: int = 700
    chunk_overlap: int = 100
    # URL de Redis (estado de los trabajos, cola y checkpoints de LangGraph)
    redis_url: str = "redis://localhost:6379"
    # Workers que levanta la propia API como tareas asyncio (0 = los workers corren en otro proceso)
    workers_embebidos: int = 2
    # Trabajos que procesa en paralelo cada proceso worker (python -m app.worker)
    worker_concurrencia: int = 5
    # Tiempo máximo de un trabajo: al superarlo pasa a FAILED en vez de dejar al cliente esperando
    job_timeout_s: int = 300
    # Cuánto viven en Redis el estado de un trabajo y sus checkpoints
    job_ttl_s: int = 86400
    # Endpoint de Phoenix al que se envían las trazas ("" = observabilidad desactivada)
    phoenix_endpoint: str = ""
    # Nombre del proyecto en Phoenix (sin valor por defecto en el código: sale de PHOENIX_PROJECT_NAME y es obligatorio si hay endpoint)
    phoenix_proyecto: str = ""


# Lee una variable de entorno entera y valida que no sea menor al mínimo permitido
def _leer_entero(nombre: str, por_defecto: int, minimo: int) -> int:
    # Lee la variable como texto; si no existe usa el valor por defecto
    crudo = os.getenv(nombre, "").strip() or str(por_defecto)
    # Intenta convertir el texto a entero
    try:
        valor = int(crudo)
    # Si no es un número, avisa con un mensaje claro en vez de dejar un error críptico
    except ValueError as error:
        raise ConfigError(f"{nombre} debe ser un número entero, pero vale '{crudo}'.") from error
    # Valida el mínimo permitido
    if valor < minimo:
        raise ConfigError(f"{nombre} debe ser como mínimo {minimo}, pero vale {valor}.")
    # Devuelve el valor ya validado
    return valor


# Lee una variable de entorno entera obligatoria (la dimensión de los embeddings depende del modelo, así que no tiene un valor por defecto razonable)
def _leer_entero_obligatorio(nombre: str, minimo: int) -> int:
    _leer_obligatoria(nombre)
    return _leer_entero(nombre, por_defecto=minimo, minimo=minimo)


# Lee una variable de entorno obligatoria: si falta o quedó el valor de ejemplo del .env.example, corta con un mensaje claro
def _leer_obligatoria(nombre: str) -> str:
    valor = os.getenv(nombre, "").strip()
    if not valor or valor.startswith("tu_"):
        raise ConfigError(f"Falta {nombre}. Copiá .env.example a .env y completalo.")
    return valor


# Carga y valida toda la configuración; si algo falta, corta con un ConfigError explicativo
def cargar_configuracion() -> Settings:
    # Carga el archivo .env (si existe) a las variables de entorno del proceso
    load_dotenv()

    # Lee el proveedor elegido (por defecto anthropic) y lo normaliza
    proveedor_crudo = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    # Valida que sea uno de los proveedores soportados
    if proveedor_crudo not in get_args(Proveedor):
        raise ConfigError(
            f"LLM_PROVIDER debe ser uno de {list(get_args(Proveedor))}, pero vale '{proveedor_crudo}'."
        )
    # Le dice al tipado estático que el texto ya fue validado como Proveedor
    proveedor = cast(Proveedor, proveedor_crudo)

    # Lee la API key del proveedor elegido y el modelo (no hay modelo por defecto en el código: se define en el .env)
    api_key = _leer_obligatoria(VARIABLE_API_KEY[proveedor])
    modelo = _leer_obligatoria("LLM_MODEL")

    # Lee los topes del grafo para validar que sean coherentes entre sí
    limite_recursion = _leer_entero("RECURSION_LIMIT", por_defecto=40, minimo=4)
    max_pasos_supervisor = _leer_entero("MAX_PASOS_SUPERVISOR", por_defecto=8, minimo=2)
    # Cada decisión del supervisor puede usar hasta 3 pasos del grafo (supervisor, especialista y validación): con menos, el grafo se cortaría antes de poder cerrar
    if limite_recursion < 3 * max_pasos_supervisor:
        raise ConfigError(
            f"RECURSION_LIMIT vale {limite_recursion}, pero con MAX_PASOS_SUPERVISOR={max_pasos_supervisor} necesita ser como mínimo {3 * max_pasos_supervisor}. "
            "Revisá tu .env."
        )

    # Valida que el solapamiento de los fragmentos sea menor que su tamaño (si no, el troceado no avanzaría nunca)
    chunk_size = _leer_entero("CHUNK_SIZE", por_defecto=700, minimo=100)
    chunk_overlap = _leer_entero("CHUNK_OVERLAP", por_defecto=100, minimo=0)
    if chunk_overlap >= chunk_size:
        raise ConfigError(f"CHUNK_OVERLAP ({chunk_overlap}) debe ser menor que CHUNK_SIZE ({chunk_size}).")

    # Arma y devuelve la configuración completa
    # Lee el endpoint de Phoenix; vacío significa observabilidad desactivada
    phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "").strip()
    # Con observabilidad activa el nombre del proyecto es obligatorio: no se escribe ningún nombre en el código
    phoenix_proyecto = os.getenv("PHOENIX_PROJECT_NAME", "").strip()
    # Si hay endpoint pero falta el nombre del proyecto, corta con un mensaje claro
    if phoenix_endpoint and not phoenix_proyecto:
        raise ConfigError("Falta PHOENIX_PROJECT_NAME (obligatorio cuando PHOENIX_COLLECTOR_ENDPOINT está definido). Copiá .env.example a .env y completalo.")

    return Settings(
        proveedor=proveedor,
        api_key=api_key,
        modelo=modelo,
        limite_recursion=limite_recursion,
        max_pasos_supervisor=max_pasos_supervisor,
        max_refinamientos=_leer_entero("MAX_REFINAMIENTOS", por_defecto=1, minimo=0),
        limite_iteraciones_agente=_leer_entero("LIMITE_ITERACIONES_AGENTE", por_defecto=8, minimo=2),
        pinecone_api_key=_leer_obligatoria("PINECONE_API_KEY"),
        pinecone_index=_leer_obligatoria("PINECONE_INDEX_NAME"),
        pinecone_namespace=_leer_obligatoria("PINECONE_NAMESPACE"),
        pinecone_cloud=_leer_obligatoria("PINECONE_CLOUD"),
        pinecone_region=_leer_obligatoria("PINECONE_REGION"),
        embedding_model=_leer_obligatoria("EMBEDDING_MODEL"),
        embedding_dim=_leer_entero_obligatorio("EMBEDDING_DIM", minimo=1),
        datos_dir=os.getenv("DATA_DIR", "").strip() or "data",
        top_k=_leer_entero("TOP_K", por_defecto=4, minimo=1),
        rrf_k=_leer_entero("RRF_K", por_defecto=60, minimo=1),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        redis_url=os.getenv("REDIS_URL", "").strip() or "redis://localhost:6379",
        workers_embebidos=_leer_entero("WORKERS_EMBEBIDOS", por_defecto=2, minimo=0),
        worker_concurrencia=_leer_entero("WORKER_CONCURRENCIA", por_defecto=5, minimo=1),
        job_timeout_s=_leer_entero("JOB_TIMEOUT_SEGUNDOS", por_defecto=300, minimo=5),
        job_ttl_s=_leer_entero("JOB_TTL_SEGUNDOS", por_defecto=86400, minimo=60),
        phoenix_endpoint=phoenix_endpoint,
        phoenix_proyecto=phoenix_proyecto,
    )
