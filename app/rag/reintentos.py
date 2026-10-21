# Módulo de reintentos: repite una llamada de red a Pinecone cuando la falla es transitoria (timeout, corte de conexión, 429 o 5xx)
# Cada reintento deja un log con el intento y el motivo (criterio que pidió la profesora en la Pre-entrega 2 y que se aplicó también en la Pre-entrega 4)

# Importa asyncio para esperar entre reintentos sin bloquear el event loop
import asyncio
# Importa logging para dejar registro de cada reintento
import logging
# Importa Awaitable y Callable para tipar la operación a repetir
from collections.abc import Awaitable, Callable
# Importa TypeVar para devolver el mismo tipo que devuelve la operación
from typing import TypeVar

# Importa el error de conexión de aiohttp (el SDK asíncrono de Pinecone usa aiohttp por debajo)
from aiohttp import ClientError
# Importa los errores de la API de Pinecone
from pinecone.exceptions import PineconeApiException, ServiceException

# Crea el logger del módulo
logger = logging.getLogger(__name__)

# Tipo del resultado de la operación
T = TypeVar("T")

# Códigos HTTP que vale la pena reintentar: tiempo agotado, demasiadas peticiones y errores del servidor
CODIGOS_TRANSITORIOS = frozenset({408, 429, 500, 502, 503, 504})


# Indica si un error es transitorio (reintentar puede resolverlo) o permanente (una API key inválida no se arregla reintentando)
def es_transitorio(error: BaseException) -> bool:
    if isinstance(error, (TimeoutError, ClientError, ServiceException)):
        return True
    return isinstance(error, PineconeApiException) and error.status in CODIGOS_TRANSITORIOS


# Ejecuta la operación y, si falla por algo transitorio, la repite con espera exponencial (0.5 s, 1 s, 2 s...)
async def con_reintentos(
    operacion: Callable[[], Awaitable[T]],
    descripcion: str,
    intentos: int = 3,
    espera_base_s: float = 0.5,
) -> T:
    for intento in range(1, intentos + 1):
        try:
            return await operacion()
        except Exception as error:
            # Un error permanente o el último intento: se propaga tal cual
            if not es_transitorio(error) or intento == intentos:
                if intento == intentos and es_transitorio(error):
                    logger.error("Fallaron los %s intentos de '%s': %s", intentos, descripcion, error)
                raise
            espera = espera_base_s * 2 ** (intento - 1)
            logger.warning("'%s' falló (intento %s de %s): %s. Reintenta en %.1f s", descripcion, intento, intentos, type(error).__name__, espera)
            await asyncio.sleep(espera)
    # Inalcanzable (el bucle siempre devuelve o lanza), pero el tipado lo exige
    raise RuntimeError("con_reintentos terminó sin resultado")
