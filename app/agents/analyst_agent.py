# Agente de Análisis/Cómputo: hace los cálculos que pide la pregunta (descuentos, envío, vacaciones, plazos) con herramientas deterministas
# Herramientas acotadas (mínimo privilegio): solo calcula; no busca información. Los números los producen las herramientas, nunca el LLM

# Importa logging para dejar registro cuando algo falla
import logging
# Importa Any para tipar
from typing import Any

# Importa BaseChatModel para tipar el LLM que recibe el especialista
from langchain_core.language_models.chat_models import BaseChatModel
# Importa los tipos de mensajes que se leen para armar el resultado
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
# Importa BaseTool y StructuredTool para definir las herramientas
from langchain_core.tools import BaseTool, StructuredTool
# Importa BaseModel y ValidationError para validar el resultado de cada herramienta
from pydantic import BaseModel, ValidationError

# Importa el constructor y el ejecutor de agentes ReAct
from app.agents.react import ErrorAgente, Nodo, crear_agente_react, ejecutar_agente
# Importa el registro de cálculos (contratos de entrada y salida de cada herramienta)
from app.calculos import CALCULOS
# Importa la configuración
from app.config import Settings
# Importa el estado compartido
from app.state import CalculoRealizado, EstadoOrquestador
# Importa utilidades de traza
from app.traza import extraer_pasos, extraer_texto

# Crea el logger del módulo
logger = logging.getLogger(__name__)

# Instrucciones fijas del rol
PROMPT_ANALISTA = (
    "Sos el Analista de un equipo que responde preguntas sobre las políticas de EmpresaFlomat. "
    "Recibís una orden y los fragmentos de política que otro especialista ya recuperó. "
    "Tu único trabajo es hacer los cálculos que la orden pide, SIEMPRE con tus herramientas (nunca de cabeza): "
    "calcular_descuento (descuento por cantidad y mayorista), calcular_total_pedido (total con descuento), calcular_envio (costo de envío), "
    "calcular_dias_vacaciones (días según antigüedad) y verificar_plazo_devolucion (plazo de devolución). "
    "Para un pedido encadená los cálculos: primero calcular_descuento; después calcular_total_pedido con el descuento_total_pct que devolvió; "
    "y por último calcular_envio con el total_productos que devolvió (el envío se decide sobre el monto ya con descuento). "
    "Usá exactamente los números de la orden y los que devuelven las herramientas: nunca inventes ni redondees cifras. "
    "Si falta un dato imprescindible (por ejemplo el precio unitario o la cantidad), NO lo supongas: terminá sin llamar herramientas y explicá qué dato falta. "
    "No busques información nueva. "
    "Cuando termines, respondé con 2 o 3 frases de conclusiones apoyadas en los números."
)

# Descripción de cada herramienta (es lo que lee el LLM para decidir cuál usar y cómo)
DESCRIPCIONES: dict[str, str] = {
    "calcular_descuento": (
        "Calcula el descuento por volumen (según las unidades del MISMO producto) más el 3% adicional si el cliente es mayorista. "
        "Devuelve un JSON con descuento_volumen_pct, descuento_mayorista_pct y descuento_total_pct."
    ),
    "calcular_total_pedido": (
        "Calcula el total de los productos aplicando un porcentaje de descuento al precio de lista. "
        "Usar el descuento_total_pct que devolvió calcular_descuento. Devuelve un JSON con subtotal, ahorro y total_productos."
    ),
    "calcular_envio": (
        "Calcula el costo de envío según el monto de la compra (productos ya con descuento, sin envío). "
        "Devuelve un JSON con envio_gratis, costo_envio y total_con_envio."
    ),
    "calcular_dias_vacaciones": (
        "Calcula los días corridos de vacaciones anuales de un empleado según sus años de antigüedad. "
        "Devuelve un JSON con dias_corridos, tramo y nota."
    ),
    "verificar_plazo_devolucion": (
        "Verifica si una devolución está dentro del plazo de 10 días corridos desde la recepción (o si se acepta por defecto de fábrica). "
        "Devuelve un JSON con dentro_de_plazo, devolucion_aceptada, plazo_dias y motivo."
    ),
}


# Crea una herramienta a partir de un cálculo del registro: entrada validada por su contrato Pydantic y salida validada por el contrato de salida
def _crear_herramienta(nombre: str) -> BaseTool:
    calculo = CALCULOS[nombre]

    # Ejecuta el cálculo: reconstruye la entrada (ya validada por LangChain), calcula y devuelve la salida validada en JSON
    async def ejecutar(**argumentos: Any) -> str:
        salida: BaseModel = calculo.funcion(calculo.entrada.model_validate(argumentos))
        return salida.model_dump_json()

    return StructuredTool.from_function(coroutine=ejecutar, name=nombre, description=DESCRIPCIONES[nombre], args_schema=calculo.entrada)


# Lista de herramientas del analista (una por cálculo del registro)
HERRAMIENTAS_ANALISTA: list[BaseTool] = [_crear_herramienta(nombre) for nombre in CALCULOS]


# Arma la instrucción del analista: la orden del supervisor + SOLO los fragmentos que necesita (nada más del estado)
def armar_instruccion_analista(instruccion: str, investigacion: dict[str, Any]) -> str:
    # Un bloque por fragmento: fuente, sección y texto
    bloques = [f"- [{f['fuente']} · {f['seccion']}] {f['texto']}" for f in investigacion.get("fragmentos", [])]
    # Une la orden con los datos
    return f"{instruccion}\n\nFragmentos de política recuperados:\n" + "\n".join(bloques)


# Arma el resultado del análisis SOLO con lo que devolvieron las herramientas (los números no los inventa el LLM)
def construir_analisis(mensajes: list[BaseMessage]) -> dict[str, Any]:
    # Argumentos con los que el LLM llamó a cada herramienta, por id de llamada
    llamadas: dict[str, dict[str, Any]] = {}
    for mensaje in mensajes:
        if isinstance(mensaje, AIMessage):
            for llamada in mensaje.tool_calls:
                llamadas[llamada["id"] or ""] = llamada["args"]
    calculos: list[dict[str, Any]] = []
    # Recorre los resultados de herramientas
    for mensaje in mensajes:
        # Ignora todo lo que no sea el resultado de una herramienta
        if not isinstance(mensaje, ToolMessage) or mensaje.name not in CALCULOS:
            continue
        # Ignora las llamadas que fallaron (argumentos inválidos): el LLM las ve y las corrige
        if mensaje.status == "error":
            continue
        # Valida el resultado con el contrato de salida de esa herramienta y normaliza las entradas con el contrato de entrada (tipos y valores por defecto)
        calculo = CALCULOS[mensaje.name]
        try:
            resultado = calculo.salida.model_validate_json(extraer_texto(mensaje)).model_dump()
            entradas = calculo.entrada.model_validate(llamadas.get(mensaje.tool_call_id, {})).model_dump()
        except ValidationError:
            continue
        calculos.append(
            CalculoRealizado(
                herramienta=mensaje.name,  # type: ignore[arg-type]
                entradas=entradas,
                resultado=resultado,
                politica=resultado["politica"],
            ).model_dump()
        )
    return {"calculos": calculos, "conclusiones": extraer_texto(mensajes[-1]).strip() if mensajes else ""}


# Crea el nodo del analista para el grafo principal
def crear_nodo_analista(llm: BaseChatModel, configuracion: Settings) -> Nodo:
    # Crea el agente ReAct del analista
    agente = crear_agente_react(llm, HERRAMIENTAS_ANALISTA, PROMPT_ANALISTA)

    # Nodo: recibe el estado, trabaja con la instrucción + los fragmentos y devuelve solo sus campos
    async def nodo_analista(state: EstadoOrquestador) -> dict[str, Any]:
        # Cuenta este intento
        intento = state.get("intentos_analista", 0) + 1
        # Arma lo que ve el analista: la orden y los fragmentos (no ve la conversación ni las decisiones del supervisor)
        instruccion = armar_instruccion_analista(state["instruccion"], state.get("investigacion", {}))
        # Intenta correr al especialista
        try:
            mensajes = await ejecutar_agente(agente, instruccion, configuracion.limite_iteraciones_agente)
            analisis = construir_analisis(mensajes)
            pasos = extraer_pasos(mensajes[1:])
            error = ""
            resumen = f"hizo {len(analisis['calculos'])} cálculo(s)"
        # Si falla, lo registra y deja el análisis vacío (la validación lo va a rechazar)
        except ErrorAgente as falla:
            logger.warning("El analista falló: %s", falla)
            analisis, pasos, error, resumen = {}, [], str(falla), "no pudo completar el análisis"
        # Devuelve solo sus campos + su aporte al registro de contribuciones
        return {
            "analisis": analisis,
            "intentos_analista": intento,
            "ultimo_agente": "analista",
            "contribuciones": [{"agente": "analista", "intento": intento, "resumen": resumen, "pasos": pasos, "error": error}],
        }

    return nodo_analista
