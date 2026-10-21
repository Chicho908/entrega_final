# Agente de Investigación: recupera fragmentos de las políticas de EmpresaFlomat con el RAG híbrido (BM25 + vectores en Pinecone)
# Herramientas acotadas (mínimo privilegio): solo puede BUSCAR en las políticas; no calcula ni redacta la respuesta final

# Importa logging para dejar registro cuando algo falla
import logging
# Importa Any y Literal para tipar
from typing import Any, Literal

# Importa BaseChatModel para tipar el LLM que recibe el especialista
from langchain_core.language_models.chat_models import BaseChatModel
# Importa los tipos de mensajes que se leen para armar el resultado
from langchain_core.messages import BaseMessage, ToolMessage
# Importa BaseTool y tool para definir la herramienta
from langchain_core.tools import BaseTool, ToolException, tool
# Importa BaseModel, Field y ValidationError para validar la entrada y la salida de la herramienta
from pydantic import BaseModel, Field, ValidationError

# Importa el constructor y el ejecutor de agentes ReAct
from app.agents.react import ErrorAgente, Nodo, crear_agente_react, ejecutar_agente
# Importa la configuración
from app.config import Settings
# Importa la categoría de política (filtro por metadata)
from app.rag.corpus import Categoria
# Importa el contrato del recuperador y el resultado de búsqueda
from app.rag.recuperador import Recuperador
# Importa el estado compartido y el fragmento de política
from app.state import EstadoOrquestador, FragmentoPolitica
# Importa utilidades de traza
from app.traza import extraer_pasos, extraer_texto

# Crea el logger del módulo
logger = logging.getLogger(__name__)

# Prefijo con el que se reconoce una falla de infraestructura de la fuente (distinta de "no hay datos")
PREFIJO_FUENTE_CAIDA = "FUENTE_CAIDA"

# Instrucciones fijas del rol (el LLM del investigador solo conoce este rol, no el resto del sistema)
PROMPT_INVESTIGADOR = (
    "Sos el Investigador de un equipo que responde preguntas sobre las políticas internas de EmpresaFlomat: "
    "envíos, devoluciones, descuentos por volumen y vacaciones. "
    "Tu único trabajo es recuperar los fragmentos de política relevantes con tu herramienta buscar_politicas (búsqueda híbrida: palabras clave + semántica). "
    "Nunca inventes datos: solo podés informar lo que la herramienta devuelva. "
    "Hacé una búsqueda por cada tema distinto de la pregunta (por ejemplo, una por descuentos y otra por envíos) y escribí la consulta con las palabras de la política, no con los datos del usuario. "
    "Si sabés de qué política se trata, filtrá con el parámetro categoria. "
    "Si una búsqueda devuelve estado 'sin_resultados', probá una vez más con otras palabras del mismo tema. "
    "Si la herramienta falla con FUENTE_CAIDA, es una falla técnica (no falta de datos): reintentá UNA vez y, si vuelve a fallar, terminá y explicá que la fuente no respondió. "
    "Si los fragmentos recuperados no tratan el tema pedido, NO los reinterpretes ni los fuerces: terminá y explicá que las políticas no tienen información sobre eso. "
    "No calcules nada: eso lo hace otro especialista. "
    "Cuando termines, respondé con un resumen de 1 o 2 frases de lo que encontraste (qué políticas y qué dice cada una)."
)


# Esquema de entrada de buscar_politicas
class BuscarInput(BaseModel):
    # Consulta en lenguaje natural o con palabras clave
    consulta: str = Field(min_length=2, max_length=300, description="Qué buscar, por ejemplo 'costo de envío para compras menores a 50.000'.")
    # Filtro opcional por metadata: restringe la búsqueda a una política
    categoria: Categoria | None = Field(default=None, description="Política donde buscar: descuentos, devoluciones, envios o vacaciones. Omitir si no se sabe.")
    # Cantidad máxima de fragmentos
    max_resultados: int | None = Field(default=None, ge=1, le=8, description="Cantidad máxima de fragmentos (1 a 8). Omitir para usar el valor por defecto.")


# Esquema de salida de buscar_politicas (también validado: lo que devuelve la herramienta tiene que cumplir el contrato)
class SalidaBusqueda(BaseModel):
    # Consulta que se ejecutó
    consulta: str
    # "ok" si hay fragmentos; "sin_resultados" si la búsqueda funcionó pero no encontró nada
    estado: Literal["ok", "sin_resultados"]
    # Cómo se resolvió la búsqueda (hibrido o solo_bm25 si Pinecone no respondió)
    modo: Literal["hibrido", "solo_bm25"] | None = None
    # Fragmentos recuperados con su procedencia
    resultados: list[FragmentoPolitica]
    # Orientación para el LLM cuando no hay resultados
    mensaje: str = ""


# Crea la herramienta de búsqueda ligada a un recuperador (en producción el híbrido; en los tests uno falso)
def crear_herramienta_busqueda(recuperador: Recuperador, top_k: int) -> BaseTool:
    # Herramienta del investigador: búsqueda híbrida en las políticas
    @tool(args_schema=BuscarInput)
    async def buscar_politicas(consulta: str, categoria: Categoria | None = None, max_resultados: int | None = None) -> str:
        """Busca en las políticas de EmpresaFlomat (envíos, devoluciones, descuentos, vacaciones) con búsqueda híbrida.
        Devuelve un JSON con estado ('ok' o 'sin_resultados') y los fragmentos más relevantes
        (fragment_id, fuente, pagina, categoria, seccion, texto y score). Si la fuente no responde, falla con un error FUENTE_CAIDA."""
        # Intenta consultar la fuente
        try:
            resultado = await recuperador.buscar(consulta, max_resultados or top_k, categoria)
        # Falla de infraestructura: se lanza como excepción (no como texto) para que el span de la herramienta quede en ERROR en el dashboard
        except Exception as error:
            logger.exception("Falla al buscar '%s'", consulta)
            raise ToolException(f"{PREFIJO_FUENTE_CAIDA}: la búsqueda en las políticas no respondió ({type(error).__name__}). Es una falla técnica, no falta de datos.") from error
        # Sin resultados NO es un error: la fuente funcionó y simplemente no hay datos (el span queda en OK)
        if not resultado.fragmentos:
            salida = SalidaBusqueda(
                consulta=consulta,
                estado="sin_resultados",
                modo=resultado.modo,
                resultados=[],
                mensaje="Ningún fragmento coincide. Probá con otras palabras del mismo tema o sin filtrar por categoría. Las políticas solo cubren envíos, devoluciones, descuentos y vacaciones.",
            )
        else:
            salida = SalidaBusqueda(consulta=consulta, estado="ok", modo=resultado.modo, resultados=resultado.fragmentos)
        # Devuelve el resultado ya validado con Pydantic, serializado como JSON
        return salida.model_dump_json()

    return buscar_politicas


# Arma el resultado de la investigación SOLO con lo que devolvieron las herramientas (el LLM no puede inventar fragmentos)
# "previos" son los fragmentos de una investigación anterior ya validada: un refinamiento los COMPLETA en vez de pisarlos
def construir_investigacion(instruccion: str, mensajes: list[BaseMessage], previos: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    # Fragmentos por id (si dos búsquedas devuelven el mismo, se conserva el de mayor puntaje)
    fragmentos: dict[str, FragmentoPolitica] = {}
    for previo in previos or []:
        fragmento = FragmentoPolitica.model_validate(previo)
        fragmentos[fragmento.fragment_id] = fragmento
    # Marca si alguna búsqueda falló por la infraestructura de la fuente
    fuente_caida = False
    # Recorre los resultados de herramientas
    for mensaje in mensajes:
        # Ignora todo lo que no sea el resultado de una herramienta
        if not isinstance(mensaje, ToolMessage):
            continue
        # Lee el texto devuelto por la herramienta
        contenido = extraer_texto(mensaje)
        # Una herramienta que falló no aporta fragmentos; si fue por la fuente, queda registrado
        if mensaje.status == "error":
            fuente_caida = fuente_caida or contenido.startswith(PREFIJO_FUENTE_CAIDA)
            continue
        # Valida la salida con su contrato: lo que no cumple se descarta
        try:
            salida = SalidaBusqueda.model_validate_json(contenido)
        except ValidationError:
            continue
        # Guarda cada fragmento encontrado
        for fragmento in salida.resultados:
            existente = fragmentos.get(fragmento.fragment_id)
            if existente is None or fragmento.score > existente.score:
                fragmentos[fragmento.fragment_id] = fragmento
    # El resumen es la última respuesta de texto del LLM
    resumen = extraer_texto(mensajes[-1]).strip() if mensajes else ""
    # Devuelve el diccionario crudo: la validación decide después si cumple el contrato
    return {
        "consulta": instruccion,
        "fragmentos": [f.model_dump() for f in sorted(fragmentos.values(), key=lambda f: (-f.score, f.fragment_id))],
        "resumen": resumen,
        "fuente_caida": fuente_caida,
    }


# Crea el nodo del investigador para el grafo principal
def crear_nodo_investigador(llm: BaseChatModel, configuracion: Settings, recuperador: Recuperador) -> Nodo:
    # Crea el agente ReAct del investigador (una sola vez, no en cada ejecución)
    agente = crear_agente_react(llm, [crear_herramienta_busqueda(recuperador, configuracion.top_k)], PROMPT_INVESTIGADOR)

    # Nodo: recibe el estado, trabaja con SOLO la instrucción y devuelve únicamente los campos que le pertenecen
    async def nodo_investigador(state: EstadoOrquestador) -> dict[str, Any]:
        # Cuenta este intento
        intento = state.get("intentos_investigador", 0) + 1
        # La instrucción específica que dejó el supervisor (el investigador NO ve el historial de la conversación)
        instruccion = state["instruccion"]
        # Intenta correr al especialista
        try:
            mensajes = await ejecutar_agente(agente, instruccion, configuracion.limite_iteraciones_agente)
            # Si la investigación anterior fue válida, esta la completa; si fue rechazada, empieza de cero
            previos = state["investigacion"]["fragmentos"] if state.get("investigacion_valida") else []
            investigacion = construir_investigacion(instruccion, mensajes, previos)
            pasos = extraer_pasos(mensajes[1:])
            error = ""
            resumen = f"recuperó {len(investigacion['fragmentos'])} fragmento(s)"
            if investigacion["fuente_caida"] and not investigacion["fragmentos"]:
                resumen = "la búsqueda en las políticas no respondió"
        # Si falla, lo registra y deja la investigación vacía (la validación la va a rechazar)
        except ErrorAgente as falla:
            logger.warning("El investigador falló: %s", falla)
            investigacion, pasos, error, resumen = {}, [], str(falla), "no pudo completar la búsqueda"
        # Devuelve solo sus campos + su aporte al registro de contribuciones
        return {
            "investigacion": investigacion,
            "intentos_investigador": intento,
            "ultimo_agente": "investigador",
            "contribuciones": [{"agente": "investigador", "intento": intento, "resumen": resumen, "pasos": pasos, "error": error}],
        }

    return nodo_investigador
