# Módulo de cadenas: la cadena LCEL de la Pre-entrega 2, ahora al servicio del sintetizador
# prompt | modelo con salida estructurada | reintentos  →  devuelve SIEMPRE un objeto RespuestaFinal validado por Pydantic (nunca texto suelto)

# Importa json para pasarle los datos al LLM en formato compacto
import json
# Importa logging para dejar registro del procesamiento y de los fallos tras los reintentos
import logging
# Importa Any para tipar los datos
from typing import Any

# Importa BaseChatModel para tipar el LLM
from langchain_core.language_models.chat_models import BaseChatModel
# Importa el error de parseo de salida estructurada (junto con ValidationError son los fallos que vale la pena reintentar)
from langchain_core.exceptions import OutputParserException
# Importa ChatPromptTemplate: el prompt modular con roles separados (sistema / humano)
from langchain_core.prompts import ChatPromptTemplate
# Importa Runnable: el tipo de una cadena LCEL
from langchain_core.runnables import Runnable
# Importa BaseModel, Field y ValidationError para el contrato de salida
from pydantic import BaseModel, Field, ValidationError

# Crea el logger del módulo
logger = logging.getLogger(__name__)

# Instrucciones fijas del rol de síntesis (sin llaves: el prompt template las interpretaría como variables)
PROMPT_SINTETIZADOR = (
    "Sos el Sintetizador de un equipo que responde preguntas sobre las políticas internas de EmpresaFlomat. "
    "Redactá la respuesta final para el usuario usando EXCLUSIVAMENTE los datos validados que te pasan: fragmentos de política (con su fuente y página) y cálculos ya hechos. "
    "Los números de los cálculos son exactos: citalos tal cual, sin recalcular ni redondear. Los importes van con signo $ y puntos de miles. "
    "Citá la fuente de cada dato con el nombre del PDF, por ejemplo politica_envios.pdf. "
    "Si los fragmentos no contienen la respuesta, o falta un dato para calcular, decilo con claridad, marcá respondida en falso y no inventes políticas ni cifras. "
    "Si hay LIMITACIONES, explicalas y no completes lo que falta. "
    "Respondé en español, en un párrafo corto o una lista breve."
)


# Contrato de salida de la cadena: lo que el LLM está obligado a devolver, ya validado
class RespuestaFinal(BaseModel):
    # Texto de la respuesta para el usuario (min_length evita una respuesta vacía o de una palabra)
    respuesta: str = Field(min_length=10, description="Respuesta final para el usuario, en español.")
    # PDF que sustentan la respuesta (se comparan después contra los que realmente se recuperaron)
    fuentes_citadas: list[str] = Field(description="Nombres de los PDF que sustentan la respuesta, por ejemplo ['politica_envios.pdf']. Lista vacía si no hay información.")
    # True si las políticas alcanzaron para responder; False si faltó información o un dato
    respondida: bool = Field(description="True si los datos alcanzaron para responder la pregunta; False si falta información o un dato.")


# Prompt modular: el rol queda separado del mensaje humano con los datos
PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", PROMPT_SINTETIZADOR),
        ("human", "Datos validados (JSON):\n{datos}"),
    ]
)


# Arma la cadena: prompt | modelo con salida estructurada | reintentos
def crear_cadena_sintesis(llm: BaseChatModel) -> Runnable[dict[str, str], Any]:
    # with_structured_output fuerza al modelo a devolver una instancia de RespuestaFinal validada por Pydantic
    modelo_estructurado = llm.with_structured_output(RespuestaFinal)
    # with_retry reintenta SOLO si la salida vino mal formada o incompleta (no reintenta una API key inválida ni una caída de red, que ya reintenta el cliente)
    return (PROMPT | modelo_estructurado).with_retry(
        retry_if_exception_type=(ValidationError, OutputParserException),
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )


# Ejecuta la cadena de forma asíncrona y deja logs del proceso, incluido el fallo definitivo tras los reintentos
async def sintetizar(cadena: Runnable[dict[str, str], Any], datos: dict[str, Any]) -> RespuestaFinal:
    logger.info("Sintetizando con %s fragmento(s) y %s cálculo(s)", len(datos.get("fragmentos", [])), len(datos.get("calculos", [])))
    try:
        salida = await cadena.ainvoke({"datos": json.dumps(datos, ensure_ascii=False)})
        resultado = salida if isinstance(salida, RespuestaFinal) else RespuestaFinal.model_validate(salida)
        logger.info("Síntesis validada: respondida=%s, fuentes=%s", resultado.respondida, resultado.fuentes_citadas)
        return resultado
    except Exception as error:
        # Si with_retry ya agotó los 3 intentos, esto es el fallo definitivo (el sintetizador usa entonces su respuesta de respaldo)
        logger.error("Falló la síntesis tras los reintentos: %s", error)
        raise
