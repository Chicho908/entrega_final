# Módulo del supervisor: el "cerebro" que decide qué nodo actúa ahora
# Dos capas: (1) un LLM propone el próximo paso con salida estructurada y (2) guardarraíles de código lo corrigen si rompe las reglas
# Así se combina la flexibilidad del LLM con garantías duras: nunca hay un supervisor infinito ni se cierra sin una rúbrica cumplida

# Importa logging para dejar registro de las decisiones y de las correcciones
import logging
# Importa Mapping para tipar el estado de solo lectura
from collections.abc import Mapping
# Importa Any para tipar los diccionarios de estado
from typing import Any

# Importa BaseChatModel para tipar el LLM del supervisor
from langchain_core.language_models.chat_models import BaseChatModel
# Importa los mensajes para armar la consulta al LLM
from langchain_core.messages import HumanMessage, SystemMessage

# Importa el tipo de nodo
from app.agents.react import Nodo
# Importa la pregunta original
from app.agents.synthesis_agent import pregunta_original
# Importa la configuración
from app.config import Settings
# Importa el detector de errores de credenciales del LLM
from app.llm import es_error_de_credenciales
# Importa los contratos
from app.state import Decision, Destino, EstadoOrquestador

# Crea el logger del módulo
logger = logging.getLogger(__name__)

# Prompt del supervisor, con la rúbrica de suficiencia del dominio
PROMPT_SUPERVISOR = """Sos el Supervisor de un equipo que responde preguntas sobre las políticas internas de EmpresaFlomat: envíos, devoluciones, descuentos por volumen y vacaciones.
Tu equipo:
- investigador: recupera fragmentos de las políticas con un RAG híbrido (con su fuente y página).
- analista: hace cálculos exactos con herramientas (descuento por cantidad y mayorista, total del pedido, costo de envío, días de vacaciones según antigüedad, plazo de devolución).
- sintetizador: redacta la respuesta final. Es el cierre: después de él no se puede corregir nada.

En cada turno recibís el ESTADO ACTUAL (resumen). Dada la situación, ¿quién debe intervenir ahora o es momento de finalizar?
Elegí en "siguiente" exactamente uno de: investigador, analista, sintetizador.

Orden normal: investigador → analista (SOLO si la pregunta pide un cálculo) → sintetizador.

RÚBRICA DE SUFICIENCIA (solo elegí sintetizador si se cumple TODO):
1. La investigación está VALIDADA y recuperó las políticas que la pregunta necesita (mirá de qué PDF salen los fragmentos).
2. Si la pregunta pide un cálculo (importes, porcentajes, totales, envío para un pedido, días de vacaciones de una antigüedad concreta, si una devolución está a tiempo...), el análisis está VALIDADO.
   Si la pregunta solo consulta qué dice una política, NO hace falta el analista: pasá directo al sintetizador.
3. Los datos alcanzan para responder la pregunta.
Si un especialista fue RECHAZADO por la validación, volvé a llamarlo con una instrucción que corrija el error específico que se informa. Si el análisis fue rechazado porque falta una política, llamá al investigador para que la recupere.
Al reintentar, NO cambies el tema que pidió el usuario: si lo pedido no está en las políticas, el reintento no debe inventar otro tema.
Si el presupuesto de intentos de un especialista se agotó, elegí sintetizador: la respuesta final va a explicar la limitación.

La "instruccion" tiene que ser AUTOCONTENIDA: el especialista NO ve la conversación ni este estado. Incluí qué buscar o calcular y TODOS los datos del usuario que hagan falta (cantidades, precios, si es mayorista, antigüedad...).
Para el sintetizador dejá la instruccion vacía. En "razon" explicá en una frase por qué elegiste ese paso."""


# Arma el resumen del estado que ve el supervisor (no el historial completo: evita la contaminación de contexto)
def resumir_estado(estado: Mapping[str, Any], paso: int, max_pasos: int, max_intentos: int) -> str:
    lineas = [
        f"Pregunta del usuario: {pregunta_original(estado)}",
        f"Paso del supervisor: {paso} de {max_pasos} (al llegar al tope se fuerza el cierre).",
        "",
    ]
    # Estado de la investigación (con las políticas de donde salieron los fragmentos, para que el supervisor juzgue la cobertura)
    if estado.get("intentos_investigador", 0) == 0:
        lineas.append("Investigación: NO realizada.")
    elif estado.get("investigacion_valida"):
        fragmentos = estado["investigacion"]["fragmentos"]
        fuentes = ", ".join(sorted({f["fuente"] for f in fragmentos}))
        lineas.append(f"Investigación: VALIDADA ({len(fragmentos)} fragmentos de: {fuentes}).")
    else:
        lineas.append(f"Investigación: RECHAZADA por la validación → {estado.get('error_investigacion')}")
    lineas.append(f"Intentos del investigador: {estado.get('intentos_investigador', 0)} de {max_intentos}.")
    # Estado del análisis
    if estado.get("intentos_analista", 0) == 0:
        lineas.append("Análisis: NO realizado.")
    elif estado.get("analisis_valido"):
        lineas.append(f"Análisis: VALIDADO ({len(estado['analisis']['calculos'])} cálculos).")
    else:
        lineas.append(f"Análisis: RECHAZADO por la validación → {estado.get('error_analisis')}")
    lineas.append(f"Intentos del analista: {estado.get('intentos_analista', 0)} de {max_intentos}.")
    return "\n".join(lineas)


# Indica si la pregunta trae cifras (se usa solo como respaldo cuando el LLM del supervisor falla: una pregunta con números probablemente pide un cálculo)
def _trae_cifras(estado: Mapping[str, Any]) -> bool:
    return any(caracter.isdigit() for caracter in pregunta_original(estado))


# Decisión determinista según el estado (se usa como respaldo si el LLM falla)
def decision_por_defecto(estado: Mapping[str, Any], max_intentos: int) -> Decision:
    # Sin investigación válida y con intentos disponibles: investigar
    if not estado.get("investigacion_valida") and estado.get("intentos_investigador", 0) < max_intentos:
        return Decision(siguiente="investigador", instruccion="", razon="regla por defecto: falta una investigación válida")
    # Con investigación válida, una pregunta con cifras y sin análisis válido (y con intentos disponibles): analizar
    if (
        estado.get("investigacion_valida")
        and not estado.get("analisis_valido")
        and _trae_cifras(estado)
        and estado.get("intentos_analista", 0) < max_intentos
    ):
        return Decision(siguiente="analista", instruccion="", razon="regla por defecto: la pregunta trae cifras y falta un análisis válido")
    # En cualquier otro caso: cerrar
    return Decision(siguiente="sintetizador", instruccion="", razon="regla por defecto: no hay más pasos útiles")


# Instrucción por defecto de un especialista (si el LLM no dio una o hubo que redirigirlo)
def instruccion_por_defecto(destino: str, estado: Mapping[str, Any]) -> str:
    # El investigador recibe la pregunta original
    if destino == "investigador":
        return f"Recuperá los fragmentos de las políticas que hagan falta para responder esta pregunta: {pregunta_original(estado)}"
    # El analista recibe la pregunta original (los fragmentos se agregan en su nodo)
    if destino == "analista":
        return f"Hacé los cálculos que pide esta pregunta con los datos que da: {pregunta_original(estado)}"
    # El sintetizador no necesita instrucción
    return ""


# Aplica los guardarraíles: corrige la propuesta del LLM si rompe una regla. Devuelve (decisión final, motivo de la corrección o "")
def aplicar_reglas(propuesta: Decision, estado: Mapping[str, Any], max_intentos: int) -> tuple[Decision, str]:
    destino: Destino = propuesta.siguiente
    nota = ""
    # Contadores de intentos y validez actuales
    intentos = {"investigador": estado.get("intentos_investigador", 0), "analista": estado.get("intentos_analista", 0)}
    inv_ok = bool(estado.get("investigacion_valida"))
    ana_ok = bool(estado.get("analisis_valido"))

    # Regla 1: no se llama a un especialista que ya agotó sus intentos
    if destino in intentos and intentos[destino] >= max_intentos:
        # Si ya hay investigación válida y el análisis se intentó y falló con intentos disponibles, sigue con el análisis; si no, cierra
        destino = "analista" if (destino == "investigador" and inv_ok and intentos["analista"] > 0 and not ana_ok and intentos["analista"] < max_intentos) else "sintetizador"
        nota = f"se agotaron los intentos del especialista elegido ({max_intentos}); se redirige a {destino}"
    # Regla 2: no se analiza sin una investigación válida
    if destino == "analista" and not inv_ok:
        destino = "investigador" if intentos["investigador"] < max_intentos else "sintetizador"
        nota = f"no se puede analizar sin una investigación validada; se redirige a {destino}"
    # Regla 3: no se cierra sin una investigación validada si todavía quedan intentos de investigar
    if destino == "sintetizador":
        if not inv_ok and intentos["investigador"] < max_intentos:
            destino, nota = "investigador", "la rúbrica no se cumple: falta una investigación validada"
        # Regla 4: no se cierra si el análisis se intentó, fue rechazado y todavía quedan intentos de corregirlo
        elif inv_ok and intentos["analista"] > 0 and not ana_ok and intentos["analista"] < max_intentos:
            destino, nota = "analista", "la rúbrica no se cumple: el análisis fue rechazado y se puede corregir"

    # Si la propuesta no cambió, la deja intacta
    if destino == propuesta.siguiente:
        final = propuesta
    else:
        final = Decision(siguiente=destino, instruccion="", razon=propuesta.razon)
    # Completa la instrucción si falta y agrega el error previo cuando se repite un especialista
    instruccion = final.instruccion.strip() or instruccion_por_defecto(final.siguiente, estado)
    error_previo = {"investigador": estado.get("error_investigacion", ""), "analista": estado.get("error_analisis", "")}.get(final.siguiente, "")
    if error_previo and intentos.get(final.siguiente, 0) > 0 and error_previo not in instruccion:
        instruccion += f"\nCorregí este problema detectado por la validación en tu intento anterior: {error_previo}"
    # El sintetizador nunca lleva instrucción
    if final.siguiente == "sintetizador":
        instruccion = ""
    return final.model_copy(update={"instruccion": instruccion}), nota


# Crea el nodo supervisor para el grafo principal
def crear_nodo_supervisor(llm: BaseChatModel, configuracion: Settings) -> Nodo:
    # Intentos máximos por especialista: el primero + los refinamientos permitidos
    max_intentos = 1 + configuracion.max_refinamientos
    # LLM con salida estructurada: está obligado a devolver una Decision válida (el Literal impide nodos inventados)
    llm_decisor = llm.with_structured_output(Decision)

    # Nodo: decide el próximo paso y deja la decisión en el estado
    async def nodo_supervisor(state: EstadoOrquestador) -> dict[str, Any]:
        # Cuenta esta decisión
        paso = state.get("pasos_supervisor", 0) + 1
        propuesta_llm = "(sin consultar al LLM)"
        # Guardarraíl anti "supervisor infinito": al llegar al tope se cierra sin consultar al LLM
        if paso >= configuracion.max_pasos_supervisor:
            final = Decision(siguiente="sintetizador", instruccion="", razon="tope de pasos del supervisor alcanzado; se cierra con lo disponible")
            nota = f"tope de {configuracion.max_pasos_supervisor} pasos alcanzado"
        else:
            # Consulta al LLM con el resumen del estado (no con todo el historial)
            try:
                resumen = resumir_estado(state, paso, configuracion.max_pasos_supervisor, max_intentos)
                salida = await llm_decisor.ainvoke([SystemMessage(content=PROMPT_SUPERVISOR), HumanMessage(content=resumen)])
                propuesta = salida if isinstance(salida, Decision) else Decision.model_validate(salida)
                propuesta_llm = propuesta.siguiente
            # Si el LLM falla o devuelve algo inválido, decide la regla por defecto (el sistema no se cae)
            except Exception as error:
                # API key inválida: no se sigue con la regla por defecto, el trabajo debe terminar en FAILED
                if es_error_de_credenciales(error):
                    raise
                logger.exception("El LLM del supervisor falló; se usa la decisión por defecto")
                propuesta = decision_por_defecto(state, max_intentos)
                propuesta_llm = "(falló el LLM)"
            # Aplica los guardarraíles sobre la propuesta
            final, nota = aplicar_reglas(propuesta, state, max_intentos)
        # Registra la decisión para auditoría y traza
        registro = {
            "paso": paso,
            "propuesta": propuesta_llm,
            "final": final.siguiente,
            "razon": final.razon,
            "guardarrail": nota,
            "instruccion": final.instruccion,
        }
        logger.debug("Supervisor paso %s: %s → %s", paso, propuesta_llm, final.siguiente)
        # Devuelve solo los campos del supervisor
        return {"siguiente": final.siguiente, "instruccion": final.instruccion, "pasos_supervisor": paso, "decisiones": [registro]}

    return nodo_supervisor
