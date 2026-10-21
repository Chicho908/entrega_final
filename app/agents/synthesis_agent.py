# Agente de Síntesis: redacta la respuesta final usando SOLO los datos ya validados
# No tiene herramientas: no busca ni calcula, solo redacta (y avisa las limitaciones si algo no se pudo validar)
# Usa la cadena LCEL con salida estructurada de la Pre-entrega 2 (app/cadenas.py)

# Importa logging para dejar registro cuando falla el LLM
import logging
# Importa Mapping para tipar el estado de solo lectura
from collections.abc import Mapping
# Importa Any para tipar los diccionarios de estado
from typing import Any

# Importa BaseChatModel para tipar el LLM del sintetizador
from langchain_core.language_models.chat_models import BaseChatModel
# Importa los mensajes para armar la respuesta y leer la pregunta
from langchain_core.messages import AIMessage, HumanMessage

# Importa el tipo de nodo
from app.agents.react import Nodo
# Importa la cadena de síntesis y su contrato
from app.cadenas import RespuestaFinal, crear_cadena_sintesis, sintetizar
# Importa el detector de errores de credenciales del LLM
from app.llm import es_error_de_credenciales
# Importa el estado compartido
from app.state import EstadoOrquestador
# Importa utilidades de traza
from app.traza import extraer_texto

# Crea el logger del módulo
logger = logging.getLogger(__name__)


# Obtiene la pregunta original (el primer mensaje humano del historial)
def pregunta_original(estado: Mapping[str, Any]) -> str:
    # Recorre el historial buscando el primer mensaje del usuario
    for mensaje in estado.get("messages", []):
        if isinstance(mensaje, HumanMessage):
            return extraer_texto(mensaje)
    return ""


# Arma las limitaciones: qué partes no se pudieron validar y por qué
def armar_limitaciones(estado: Mapping[str, Any]) -> list[str]:
    limitaciones: list[str] = []
    # Investigación no válida
    if not estado.get("investigacion_valida"):
        limitaciones.append(f"No se pudo validar la investigación: {estado.get('error_investigacion') or 'no se ejecutó'}.")
    # Análisis intentado y no válido (si el analista nunca se llamó, la pregunta no pedía cálculos y no es una limitación)
    if estado.get("intentos_analista", 0) > 0 and not estado.get("analisis_valido"):
        limitaciones.append(f"No se pudo validar el análisis: {estado.get('error_analisis') or 'no se ejecutó'}.")
    return limitaciones


# Fuentes (PDF con página) de los fragmentos validados, sin repetir y en orden estable
def fuentes_disponibles(estado: Mapping[str, Any]) -> list[str]:
    if not estado.get("investigacion_valida"):
        return []
    return sorted({f["fuente"] for f in estado["investigacion"]["fragmentos"]})


# Respuesta de emergencia sin LLM: se usa si el LLM del sintetizador falla (el usuario igual recibe algo útil y honesto)
def respuesta_de_respaldo(estado: Mapping[str, Any], limitaciones: list[str]) -> str:
    partes: list[str] = []
    # Si el análisis es válido, informa los resultados de los cálculos
    if estado.get("analisis_valido"):
        for calculo in estado["analisis"]["calculos"]:
            partes.append(f"{calculo['herramienta']}: {calculo['resultado']}.")
    # Si hay investigación válida, indica de qué políticas salen los datos
    elif estado.get("investigacion_valida"):
        partes.append("Se recuperaron fragmentos de: " + ", ".join(fuentes_disponibles(estado)) + ".")
    # Siempre informa las limitaciones
    partes.extend(limitaciones)
    return " ".join(partes) or "No se pudo generar una respuesta."


# Arma los datos que ve el LLM: SOLO lo que pasó la validación (más las limitaciones)
def armar_datos(estado: Mapping[str, Any], limitaciones: list[str]) -> dict[str, Any]:
    datos: dict[str, Any] = {"pregunta": pregunta_original(estado), "limitaciones": limitaciones}
    if estado.get("investigacion_valida"):
        # Solo los fragmentos reales: el resumen en texto libre del investigador se descarta para que no contradiga a los datos
        datos["fragmentos"] = [
            {"fuente": f["fuente"], "pagina": f["pagina"], "seccion": f["seccion"], "texto": f["texto"]} for f in estado["investigacion"]["fragmentos"]
        ]
    if estado.get("analisis_valido"):
        datos["calculos"] = [{"herramienta": c["herramienta"], "entradas": c["entradas"], "resultado": c["resultado"]} for c in estado["analisis"]["calculos"]]
    return datos


# Arma el texto final: la respuesta del LLM + la línea de fuentes (solo las que de verdad se recuperaron)
def componer_respuesta(salida: RespuestaFinal, disponibles: list[str]) -> tuple[str, list[str]]:
    # Se descartan las fuentes que el LLM citó pero no existen entre las recuperadas (no se pueden citar PDF inventados)
    citadas = [fuente for fuente in dict.fromkeys(salida.fuentes_citadas) if fuente in disponibles]
    # Si respondió pero no citó ninguna válida, se usan las recuperadas (la trazabilidad no depende de que el LLM se acuerde)
    # Si NO pudo responder, no cita ninguna fuente: lo recuperado no sustenta ninguna respuesta (aunque el LLM haya nombrado alguna)
    fuentes = (citadas or disponibles) if salida.respondida else []
    texto = salida.respuesta.strip()
    if fuentes:
        texto += "\n\nFuentes: " + ", ".join(fuentes)
    return texto, fuentes


# Crea el nodo del sintetizador para el grafo principal
def crear_nodo_sintetizador(llm: BaseChatModel) -> Nodo:
    # Cadena LCEL con salida estructurada (se crea una vez, no en cada ejecución)
    cadena = crear_cadena_sintesis(llm)

    # Nodo: arma los datos validados, redacta y devuelve solo sus campos
    async def nodo_sintetizador(state: EstadoOrquestador) -> dict[str, Any]:
        # Calcula qué quedó sin validar y qué fuentes hay
        limitaciones = armar_limitaciones(state)
        disponibles = fuentes_disponibles(state)
        # Intenta redactar con la cadena
        try:
            salida = await sintetizar(cadena, armar_datos(state, limitaciones))
            texto, fuentes = componer_respuesta(salida, disponibles)
        # Si el LLM falla, usa la respuesta de respaldo en vez de romper todo el flujo
        except Exception as error:
            # API key inválida: no se disimula con una respuesta de respaldo, el trabajo debe terminar en FAILED
            if es_error_de_credenciales(error):
                raise
            logger.exception("Falla del LLM en la síntesis; se usa la respuesta de respaldo")
            texto, fuentes = respuesta_de_respaldo(state, limitaciones), disponibles
        # Devuelve su respuesta como mensaje del historial y como campo propio
        return {
            "messages": [AIMessage(content=texto)],
            "respuesta_final": texto,
            "fuentes": fuentes,
            "contribuciones": [{"agente": "sintetizador", "intento": 1, "resumen": "redactó la respuesta final", "pasos": [], "error": ""}],
        }

    return nodo_sintetizador
