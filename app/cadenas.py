# Módulo de cadenas: la cadena LCEL de la Pre-entrega 2, ahora al servicio del sintetizador
# prompt | modelo con salida estructurada  →  devuelve SIEMPRE un objeto RespuestaFinal validado por Pydantic (nunca texto suelto)
# Si la salida viene mal formada se reintenta con espera exponencial y CADA reintento deja un log (criterio de la profesora en la Pre-entrega 2)

# Importa asyncio para esperar entre reintentos sin bloquear el event loop
import asyncio
# Importa json para pasarle los datos al LLM en formato compacto
import json
# Importa logging para dejar registro del procesamiento y de los fallos tras los reintentos
import logging
# Importa random para agregar jitter (variación aleatoria) a la espera entre reintentos
import random
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

# Importa la verificación de importes de la respuesta contra los datos validados
from app.montos import montos_no_respaldados

# Crea el logger del módulo
logger = logging.getLogger(__name__)

# Intentos totales de la síntesis ante una salida mal formada y espera base entre intentos (segundos)
INTENTOS_SINTESIS = 3
ESPERA_BASE_S = 1.0

# Instrucciones fijas del rol de síntesis (sin llaves: el prompt template las interpretaría como variables)
PROMPT_SINTETIZADOR = (
    "Sos el Sintetizador de un equipo que responde preguntas sobre las políticas internas de EmpresaFlomat. "
    "Redactá la respuesta final para el usuario usando EXCLUSIVAMENTE los datos validados que te pasan: fragmentos de política (con su fuente y página) y cálculos ya hechos. "
    "Los números de los cálculos son exactos: citalos tal cual, sin recalcular ni redondear. Los importes van con signo $ y puntos de miles. "
    "NO calcules importes que no estén en los datos (por ejemplo, cuánto vale cada descuento por separado): usá solo los importes que te pasan y expresá los descuentos como porcentajes. "
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


# Error de la síntesis: la respuesta incluye importes en pesos que no figuran en los datos validados (el LLM los inventó o los calculó por su cuenta)
class MontoNoRespaldado(ValueError):
    pass


# Prompt modular: el rol queda separado del mensaje humano con los datos
PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", PROMPT_SINTETIZADOR),
        ("human", "Datos validados (JSON):\n{datos}"),
    ]
)


# Arma la cadena: prompt | modelo con salida estructurada
def crear_cadena_sintesis(llm: BaseChatModel) -> Runnable[dict[str, str], Any]:
    # with_structured_output fuerza al modelo a devolver una instancia de RespuestaFinal validada por Pydantic
    return PROMPT | llm.with_structured_output(RespuestaFinal)


# Ejecuta la cadena de forma asíncrona. Reintenta SOLO si la salida vino mal formada o incompleta (ValidationError / OutputParserException) o trae importes inventados (MontoNoRespaldado):
# una API key inválida no se arregla reintentando y las caídas de red ya las reintenta el cliente del proveedor. Cada reintento queda en el log
async def sintetizar(cadena: Runnable[dict[str, str], Any], datos: dict[str, Any]) -> RespuestaFinal:
    logger.info("Sintetizando con %s fragmento(s) y %s cálculo(s)", len(datos.get("fragmentos", [])), len(datos.get("calculos", [])))
    entrada = {"datos": json.dumps(datos, ensure_ascii=False)}
    for intento in range(1, INTENTOS_SINTESIS + 1):
        try:
            salida = await cadena.ainvoke(entrada)
            resultado = salida if isinstance(salida, RespuestaFinal) else RespuestaFinal.model_validate(salida)
            # Guardarraíl: todo importe en pesos de la respuesta tiene que salir de los datos validados
            inventados = montos_no_respaldados(resultado.respuesta, datos)
            if inventados:
                # En el reintento se le dice al LLM exactamente qué importes sobran, así no repite el error
                correccion = f"Tu respuesta anterior incluyó importes que no figuran en los datos: {inventados}. No calcules importes: usá solo los de los datos."
                entrada = {"datos": json.dumps({**datos, "correccion": correccion}, ensure_ascii=False)}
                raise MontoNoRespaldado(f"importes sin respaldo: {inventados}")
        except (ValidationError, OutputParserException, MontoNoRespaldado) as error:
            # Último intento: es el fallo definitivo (el sintetizador usa entonces su respuesta de respaldo)
            if intento == INTENTOS_SINTESIS:
                logger.error("Falló la síntesis tras %s intentos: %s", INTENTOS_SINTESIS, error)
                raise
            espera = ESPERA_BASE_S * 2 ** (intento - 1) * random.uniform(0.5, 1.5)
            logger.warning("La salida del LLM no cumplió el contrato (intento %s de %s): %s. Reintenta en %.1f s", intento, INTENTOS_SINTESIS, type(error).__name__, espera)
            await asyncio.sleep(espera)
        except Exception as error:
            # Cualquier otro error (credenciales, red agotada) no se reintenta acá
            logger.error("Falló la síntesis sin posibilidad de reintento: %s", error)
            raise
        else:
            logger.info("Síntesis validada: respondida=%s, fuentes=%s", resultado.respondida, resultado.fuentes_citadas)
            return resultado
    # Inalcanzable (el bucle siempre devuelve o lanza), pero el tipado lo exige
    raise RuntimeError("sintetizar terminó sin resultado")
