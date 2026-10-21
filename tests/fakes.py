# Dobles de prueba compartidos: un LLM falso que sigue un guion, un recuperador de prueba (el real, con embeddings y almacén en memoria),
# guiones de ejemplo y el armado del orquestador
# Las herramientas, la validación, los guardarraíles, el RAG (BM25 + fusión RRF) y el grafo son los REALES; solo cambian los "cerebros" externos (LLM y Pinecone)

# Importa asyncio para simular la demora del grafo falso
import asyncio
# Importa math y zlib para armar embeddings falsos deterministas
import math
import zlib
# Importa Sequence para tipar los mensajes que recibe el LLM falso
from collections.abc import Sequence
# Importa dataclass para definir los dobles simples
from dataclasses import dataclass
# Importa Any para tipar
from typing import Any

# Importa BaseChatModel para crear el LLM falso
from langchain_core.language_models.chat_models import BaseChatModel
# Importa los tipos de mensajes
from langchain_core.messages import AIMessage, BaseMessage
# Importa los tipos de resultado de un modelo de chat
from langchain_core.outputs import ChatGeneration, ChatResult
# Importa RunnableLambda para que la "salida estructurada" falsa sea un Runnable (se encadena con el prompt como la real)
from langchain_core.runnables import RunnableLambda
# Importa el checkpointer en memoria y el tipo base
from langgraph.checkpoint.base import BaseCheckpointSaver
# Importa Field para declarar campos con valor inicial en el LLM falso
from pydantic import Field, ValidationError

# Importa la cadena de síntesis (su contrato de salida)
from app.cadenas import RespuestaFinal
# Importa Settings para armar la configuración de prueba
from app.config import Settings
# Importa el grupo de LLM y el armado del grafo
from app.graph import LLMs, crear_grafo
# Importa el despachador
from app.hitl import Despachador
# Importa el corpus real y el recuperador real
from app.rag.corpus import Chunk, cargar_chunks
from app.rag.recuperador import Recuperador, RecuperadorHibrido, ResultadoBusqueda, tokenizar
# Importa el contrato de decisión
from app.state import Decision

# ---------- LLM falso ----------


# LLM falso: sirve para los tres usos del proyecto (chat con herramientas, salida estructurada y texto simple)
class LLMGuionado(BaseChatModel):
    # Respuestas que va devolviendo, una por llamada (AIMessage, Decision, RespuestaFinal o una Exception)
    respuestas: list[Any]
    # Si es True, repite la última respuesta indefinidamente (simula un supervisor que nunca termina)
    repetir_ultima: bool = False
    # Posición de la próxima respuesta
    indice: int = 0
    # Mensajes que recibió en cada llamada
    recibidos: list[list[BaseMessage]] = Field(default_factory=list)

    # Identifica el tipo de modelo (requisito de BaseChatModel)
    @property
    def _llm_type(self) -> str:
        return "guionado"

    # bind_tools devuelve el mismo objeto: el guion ya decide cuándo "pedir" herramientas
    def bind_tools(self, tools: Any, **kwargs: Any) -> "LLMGuionado":
        return self

    # Toma la próxima respuesta del guion (lanza la excepción si el guion pone una)
    def _siguiente(self) -> Any:
        posicion = min(self.indice, len(self.respuestas) - 1) if self.repetir_ultima else self.indice
        self.indice += 1
        respuesta = self.respuestas[posicion]
        if isinstance(respuesta, Exception):
            raise respuesta
        return respuesta

    # with_structured_output devuelve un Runnable que entrega las respuestas del guion (así se puede encadenar: prompt | modelo)
    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        # Registra lo que recibió (el prompt ya armado o una lista de mensajes) y devuelve la próxima respuesta del guion
        async def invocar(entrada: Any) -> Any:
            mensajes = entrada.to_messages() if hasattr(entrada, "to_messages") else entrada
            self.recibidos.append(list(mensajes))
            return self._siguiente()

        return RunnableLambda(invocar)

    # Devuelve la siguiente respuesta del guion como mensaje de chat
    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.recibidos.append(list(messages))
        respuesta = self._siguiente()
        # Si el guion trae un texto suelto, lo convierte en un mensaje del modelo
        if isinstance(respuesta, str):
            respuesta = AIMessage(content=respuesta)
        # Devuelve una COPIA sin id: el reducer add_messages reemplaza mensajes con el mismo id
        copia = respuesta.model_copy(deep=True)
        copia.id = None
        return ChatResult(generations=[ChatGeneration(message=copia)])


# Arma un mensaje del modelo que pide una o varias herramientas a la vez
def pedir(*llamadas: tuple[str, dict[str, Any]], prefijo: str = "c") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": nombre, "args": args, "id": f"{prefijo}{i}", "type": "tool_call"} for i, (nombre, args) in enumerate(llamadas)],
    )


# Atajo para armar una decisión del supervisor
def decidir(destino: str, instruccion: str = "", razon: str = "decisión del guion") -> Decision:
    return Decision(siguiente=destino, instruccion=instruccion, razon=razon)  # type: ignore[arg-type]


# Crea un ValidationError real de Pydantic (simula una salida estructurada mal formada del LLM)
def error_de_validacion() -> ValidationError:
    try:
        RespuestaFinal.model_validate({})
    except ValidationError as error:
        return error
    raise AssertionError("model_validate({}) debería haber fallado")


# ---------- RAG de prueba: embeddings y almacén en memoria ----------

# Dimensión de los embeddings falsos
DIMENSION_FALSA = 128


# Embedder falso: bolsa de palabras con hashing (determinista y sin red). Frases con palabras en común quedan cerca
class EmbedderFalso:
    # Cuenta las llamadas para poder comprobarlas en los tests
    def __init__(self) -> None:
        self.llamadas_consulta = 0

    # Vector normalizado de un texto
    def vector(self, texto: str) -> list[float]:
        valores = [0.0] * DIMENSION_FALSA
        for palabra in tokenizar(texto):
            valores[zlib.crc32(palabra.encode()) % DIMENSION_FALSA] += 1.0
        norma = math.sqrt(sum(v * v for v in valores)) or 1.0
        return [v / norma for v in valores]

    # Embeddings de los fragmentos
    async def embeber_documentos(self, textos: Sequence[str]) -> list[list[float]]:
        return [self.vector(texto) for texto in textos]

    # Embedding de la consulta
    async def embeber_consulta(self, texto: str) -> list[float]:
        self.llamadas_consulta += 1
        return self.vector(texto)


# Almacén vectorial en memoria con similitud coseno (reemplaza a Pinecone)
class AlmacenEnMemoria:
    # Guarda (vector, categoría) por id; "falla" permite simular un Pinecone caído
    def __init__(self, falla: Exception | None = None) -> None:
        self.datos: dict[str, tuple[list[float], str]] = {}
        self.falla = falla
        self.consultas: list[tuple[int, str | None]] = []

    # Guarda los fragmentos con sus vectores
    async def guardar(self, chunks: Sequence[Chunk], vectores: Sequence[Sequence[float]]) -> int:
        for chunk, vector in zip(chunks, vectores, strict=True):
            self.datos[chunk.chunk_id] = (list(vector), chunk.categoria)
        return len(chunks)

    # Busca por similitud coseno (los vectores ya están normalizados) con filtro opcional por categoría
    async def buscar(self, vector: Sequence[float], k: int, categoria: str | None = None) -> list[tuple[str, float]]:
        self.consultas.append((k, categoria))
        if self.falla is not None:
            raise self.falla
        puntajes = [
            (sum(a * b for a, b in zip(vector, otro, strict=True)), chunk_id)
            for chunk_id, (otro, cat) in self.datos.items()
            if categoria is None or cat == categoria
        ]
        puntajes.sort(key=lambda par: (-par[0], par[1]))
        return [(chunk_id, puntaje) for puntaje, chunk_id in puntajes[:k]]


# Corpus real (los 4 PDF de data/)
def chunks_de_prueba() -> list[Chunk]:
    return cargar_chunks("data", 700, 100)


# Arma el recuperador híbrido REAL con embeddings y almacén falsos, ya cargado con el corpus
def crear_recuperador_de_prueba(falla_vectorial: Exception | None = None) -> RecuperadorHibrido:
    chunks = chunks_de_prueba()
    embedder = EmbedderFalso()
    almacen = AlmacenEnMemoria()
    almacen.datos = {c.chunk_id: (embedder.vector(c.texto), c.categoria) for c in chunks}
    almacen.falla = falla_vectorial
    return RecuperadorHibrido(chunks, embedder, almacen)


# Recuperador intermitente: falla las primeras N veces (simula Pinecone caído) y después responde con el recuperador real
class RecuperadorIntermitente:
    # Guarda el recuperador real y cuántas veces tiene que fallar
    def __init__(self, real: RecuperadorHibrido, fallas: int) -> None:
        self.real = real
        self.fallas = fallas
        self.llamadas = 0

    # Falla mientras queden fallas pendientes; después delega
    async def buscar(self, consulta: str, k: int, categoria: str | None = None) -> ResultadoBusqueda:
        self.llamadas += 1
        if self.fallas > 0:
            self.fallas -= 1
            raise ConnectionError("Pinecone caído")
        return await self.real.buscar(consulta, k, categoria)


# Recuperador que no encuentra nada (la búsqueda funciona, pero no hay datos)
class RecuperadorVacio:
    # Siempre devuelve una lista vacía de fragmentos
    async def buscar(self, consulta: str, k: int, categoria: str | None = None) -> ResultadoBusqueda:
        return ResultadoBusqueda(fragmentos=[], modo="hibrido")


# ---------- Datos y guiones ----------

# Pregunta de ejemplo de los tests: pide un cálculo encadenado (descuento → total → envío)
PREGUNTA = "Soy cliente mayorista y compro 60 unidades de $1.200 cada una. ¿Cuánto pago en total con descuentos y envío?"
# Resultados correctos de esa pregunta: descuento 10% + 3% = 13% → subtotal 72.000 → total con descuento 62.640 → envío gratis (≥ 50.000)
TOTAL_ESPERADO = 62640.0


# Configuración de prueba (no usa API key real)
def settings_de_prueba(**cambios: Any) -> Settings:
    base: dict[str, Any] = {
        "proveedor": "openai", "api_key": "no-se-usa", "modelo": "falso", "limite_recursion": 40,
        "max_pasos_supervisor": 8, "max_refinamientos": 1, "limite_iteraciones_agente": 8, "job_timeout_s": 30,
    }
    base.update(cambios)
    return Settings(**base)


# Guion del investigador: busca la política de descuentos y la de envíos, y resume
def guion_investigador() -> list[Any]:
    return [
        pedir(
            ("buscar_politicas", {"consulta": "descuento por cantidad y mayorista", "categoria": "descuentos"}),
            ("buscar_politicas", {"consulta": "costo de envío según el monto de la compra", "categoria": "envios"}),
            prefijo="i",
        ),
        AIMessage(content="Recuperé la política de descuentos por volumen (con el adicional mayorista) y la de costo de envío."),
    ]


# Guion del analista: encadena descuento → total → envío (con los valores correctos, o con el envío calculado sobre otro monto para simular un error)
def guion_analista(monto_envio: float = TOTAL_ESPERADO) -> list[Any]:
    return [
        pedir(("calcular_descuento", {"cantidad": 60, "es_mayorista": True}), prefijo="d"),
        pedir(("calcular_total_pedido", {"precio_unitario": 1200, "cantidad": 60, "descuento_total_pct": 13}), prefijo="t"),
        pedir(("calcular_envio", {"monto_compra": monto_envio}), prefijo="e"),
        AIMessage(content="Con 60 unidades y condición de mayorista corresponde un 13% de descuento; el envío es gratis."),
    ]


# Guion del sintetizador: una respuesta estructurada que cita las dos políticas
def guion_sintetizador() -> list[Any]:
    return [
        RespuestaFinal(
            respuesta="El cliente paga $62.640 en total: 13% de descuento (10% por volumen y 3% mayorista) sobre $72.000, con envío gratis.",
            fuentes_citadas=["politica_descuentos.pdf", "politica_envios.pdf"],
            respondida=True,
        )
    ]


# Guion del supervisor del camino feliz: investigar → analizar → sintetizar
def guion_supervisor_feliz() -> list[Any]:
    return [
        decidir("investigador", "Recuperá las políticas de descuento por cantidad (con el adicional mayorista) y de costo de envío.", "primero hay que conseguir las políticas"),
        decidir("analista", "Calculá el descuento de 60 unidades para un mayorista, el total con precio unitario $1.200 y el envío.", "ya hay investigación validada; falta calcular"),
        decidir("sintetizador", "", "rúbrica cumplida: investigación y análisis validados"),
    ]


# Los 4 LLM falsos de un orquestador (tipados como LLMGuionado para poder inspeccionar lo que recibieron)
@dataclass(frozen=True)
class LLMsGuionados:
    supervisor: LLMGuionado
    investigador: LLMGuionado
    analista: LLMGuionado
    sintetizador: LLMGuionado


# Arma el grafo compilado y los 4 LLM falsos a partir de los guiones
def armar_orquestador(
    supervisor: list[Any], investigador: list[Any], analista: list[Any], sintetizador: list[Any],
    configuracion: Settings | None = None, repetir_supervisor: bool = False,
    checkpointer: BaseCheckpointSaver[str] | None = None, despachador: Despachador | None = None,
    recuperador: Recuperador | None = None,
) -> tuple[Any, LLMsGuionados]:
    # Un LLM falso por rol
    llms = LLMsGuionados(
        supervisor=LLMGuionado(respuestas=supervisor, repetir_ultima=repetir_supervisor),
        investigador=LLMGuionado(respuestas=investigador),
        analista=LLMGuionado(respuestas=analista),
        sintetizador=LLMGuionado(respuestas=sintetizador),
    )
    # Compila el grafo con la configuración de prueba y el recuperador real (con embeddings falsos)
    grupo = LLMs(supervisor=llms.supervisor, investigador=llms.investigador, analista=llms.analista, sintetizador=llms.sintetizador)
    grafo = crear_grafo(grupo, configuracion or settings_de_prueba(), recuperador or crear_recuperador_de_prueba(), despachador)
    return grafo.compile(checkpointer=checkpointer), llms


# Despachador espía: en vez de enviar nada, registra a quién se le habría enviado qué
class DespachadorEspia:
    # Crea la lista donde se acumulan los envíos registrados
    def __init__(self) -> None:
        self.enviados: list[tuple[str, str, str]] = []

    # Registra el envío en vez de hacerlo y devuelve un id correlativo
    async def enviar(self, destinatario: str, asunto: str, cuerpo: str) -> str:
        self.enviados.append((destinatario, asunto, cuerpo))
        return f"espia{len(self.enviados)}"


# Foto mínima del estado del grafo que necesita el worker (lo que devuelve aget_state)
class FotoFalsa:
    # Guarda los valores del estado; sin nodos pendientes (el grafo terminó)
    def __init__(self, valores: dict[str, Any]) -> None:
        self.next: tuple[str, ...] = ()
        self.tasks: tuple[Any, ...] = ()
        self.values = valores


# Grafo falso para probar el worker sin LLM: puede tardar, fallar o responder algo fijo
class GrafoFalso:
    # Configura cuánto tarda, si falla y qué responde; lleva la cuenta de ejecuciones y de las que corren a la vez
    def __init__(self, demora_s: float = 0.0, error: Exception | None = None, respuesta: str = "respuesta falsa") -> None:
        self.demora_s = demora_s
        self.error = error
        self.respuesta = respuesta
        self.ejecuciones = 0
        self.en_paralelo = 0
        self.maximo_en_paralelo = 0

    # Simula la ejecución del grafo: cuenta las ejecuciones en paralelo, espera, falla o termina sin emitir eventos
    async def astream(self, *args: Any, **kwargs: Any) -> Any:
        self.ejecuciones += 1
        self.en_paralelo += 1
        self.maximo_en_paralelo = max(self.maximo_en_paralelo, self.en_paralelo)
        try:
            await asyncio.sleep(self.demora_s)
            if self.error is not None:
                raise self.error
        finally:
            self.en_paralelo -= 1
        return
        yield  # convierte la función en un generador asíncrono (no emite eventos)

    # Devuelve la foto final del estado (sin pasos pendientes)
    async def aget_state(self, *args: Any, **kwargs: Any) -> FotoFalsa:
        return FotoFalsa({"respuesta_final": self.respuesta, "fuentes": ["politica_envios.pdf"], "despacho": ""})
