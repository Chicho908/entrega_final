# Módulo del LLM: crea el modelo de chat (OpenAI o Anthropic) a partir de la configuración y traduce sus errores a mensajes claros
# Es la idea de la Pre-entrega 1 (una sola interfaz para varios proveedores + errores traducidos) sobre los modelos de LangChain,
# que ya envuelven los SDK asíncronos oficiales: así todas las llamadas son async y quedan trazadas en Phoenix

# Importa anthropic y openai para reconocer sus excepciones
import anthropic
import openai

# Importa BaseChatModel para tipar el modelo que devuelve la fábrica
from langchain_core.language_models.chat_models import BaseChatModel
# Importa PineconeApiException para traducir también los errores de Pinecone
from pinecone.exceptions import PineconeApiException
# Importa SecretStr para que la API key no se imprima nunca por accidente
from pydantic import SecretStr

# Importa la configuración validada
from app.config import Settings


# Crea el LLM del proveedor elegido en la configuración (la fábrica: el resto del sistema no sabe qué proveedor hay detrás)
def crear_llm(configuracion: Settings) -> BaseChatModel:
    # Convierte la API key en un SecretStr (su repr es '**********')
    clave = SecretStr(configuracion.api_key)

    # Caso OpenAI
    if configuracion.proveedor == "openai":
        # Importa el cliente de OpenAI solo cuando se usa
        from langchain_openai import ChatOpenAI

        # temperature=0 para que las decisiones de herramientas sean lo más estables posible;
        # max_retries=3 reintenta solos los errores transitorios de la API (red, rate limit)
        return ChatOpenAI(model=configuracion.modelo, api_key=clave, temperature=0, max_retries=3)

    # Caso Anthropic
    # Importa el cliente de Anthropic solo cuando se usa
    from langchain_anthropic import ChatAnthropic

    # No se pasa temperature: algunos modelos de Anthropic rechazan ese parámetro (bug visto en la Pre-entrega 1);
    # timeout=None y stop=None son los valores por defecto, se escriben porque el tipado de langchain-anthropic los exige
    return ChatAnthropic(
        model_name=configuracion.modelo,
        api_key=clave,
        max_retries=3,
        timeout=None,
        stop=None,
    )


# Recorre la cadena de causas de un error (el original puede venir envuelto en otro)
def _cadena_de_causas(error: BaseException) -> list[BaseException]:
    causas: list[BaseException] = []
    actual: BaseException | None = error
    while actual is not None and actual not in causas:
        causas.append(actual)
        actual = actual.__cause__ or actual.__context__
    return causas


# Indica si un error del LLM es de credenciales (401/403): no se arregla reintentando ni degradando la respuesta
def es_error_de_credenciales(error: BaseException) -> bool:
    # Los clientes de OpenAI y Anthropic exponen el código HTTP en status_code
    return any(getattr(causa, "status_code", None) in (401, 403) for causa in _cadena_de_causas(error))


# Traduce un error técnico a un mensaje que entienda quien consulta la tarea (idea de _traducir_error de la Pre-entrega 1)
def traducir_error(error: BaseException) -> str:
    for causa in _cadena_de_causas(error):
        # API key inválida o sin permisos
        if isinstance(causa, (openai.AuthenticationError, anthropic.AuthenticationError)):
            return "La API key del proveedor de LLM es inválida o no tiene permisos (401/403)"
        if isinstance(causa, PineconeApiException) and causa.status in (401, 403):
            return "La API key de Pinecone es inválida o no tiene permisos (401/403)"
        # Cuota agotada o demasiadas peticiones
        if isinstance(causa, (openai.RateLimitError, anthropic.RateLimitError)):
            return "Se acabó la cuota del proveedor de LLM o hay demasiadas peticiones (429)"
        # Sin conexión con el proveedor
        if isinstance(causa, (openai.APIConnectionError, anthropic.APIConnectionError)):
            return "No se pudo conectar con el proveedor de LLM: revisar la red"
    return f"{type(error).__name__}: {error}"
