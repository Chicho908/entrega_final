# Tests del dominio: el árbitro (validación), la construcción de los resultados de los especialistas, sus herramientas con contratos Pydantic,
# los guardarraíles del supervisor y la traducción de errores. Todo es código real: no hay LLM ni red

# Importa copy para modificar resultados sin tocar el original
import copy
# Importa json para armar y leer los resultados de las herramientas
import json
# Importa httpx para armar respuestas HTTP de error
import httpx
# Importa los SDK para construir sus excepciones reales
import anthropic
import openai
# Importa Any para tipar
from typing import Any

# Importa pytest y las excepciones de Pydantic y LangChain
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import ToolException
from pinecone.exceptions import PineconeApiException
from pydantic import ValidationError

# Importa lo que se prueba
from app.agents.analyst_agent import HERRAMIENTAS_ANALISTA, armar_instruccion_analista, construir_analisis
from app.agents.research_agent import crear_herramienta_busqueda, construir_investigacion
from app.calculos import CALCULOS, recalcular
from app.llm import es_error_de_credenciales, traducir_error
from app.rag.corpus import cargar_chunks
from app.state import Decision, estado_inicial
from app.supervisor import aplicar_reglas, decision_por_defecto, resumir_estado
from app.validation import nodo_validacion, validar_analisis, validar_investigacion
# Importa los dobles de prueba
from tests.fakes import RecuperadorIntermitente, RecuperadorVacio, crear_recuperador_de_prueba

# Corpus real, para armar fragmentos auténticos
CHUNKS = cargar_chunks("data", 700, 100)


# Arma el diccionario de un fragmento real de la política indicada (como lo guarda el estado)
def fragmento(fuente: str) -> dict[str, Any]:
    chunk = next(c for c in CHUNKS if c.fuente == fuente)
    return {
        "fragment_id": chunk.chunk_id, "fuente": chunk.fuente, "pagina": chunk.pagina, "categoria": chunk.categoria,
        "seccion": chunk.seccion, "texto": chunk.texto, "score": 0.03,
    }


# Investigación válida con las políticas indicadas
def investigacion(*fuentes: str) -> dict[str, Any]:
    return {"consulta": "buscar políticas", "fragmentos": [fragmento(f) for f in fuentes], "resumen": "recuperé las políticas"}


# Arma un cálculo tal como lo guarda el analista (entradas normalizadas y resultado real)
def calculo(herramienta: str, **entradas: Any) -> dict[str, Any]:
    normalizadas = CALCULOS[herramienta].entrada.model_validate(entradas).model_dump()
    resultado = recalcular(herramienta, normalizadas)
    return {"herramienta": herramienta, "entradas": normalizadas, "resultado": resultado, "politica": resultado["politica"]}


# Análisis válido de un pedido: descuento → total → envío (60 u. de $1.200, mayorista)
def analisis_pedido() -> dict[str, Any]:
    return {
        "calculos": [
            calculo("calcular_descuento", cantidad=60, es_mayorista=True),
            calculo("calcular_total_pedido", precio_unitario=1200, cantidad=60, descuento_total_pct=13),
            calculo("calcular_envio", monto_compra=62640),
        ],
        "conclusiones": "13% de descuento y envío gratis.",
    }


# ---------- Validación de la investigación ----------


# Test 1: una investigación con fragmentos reales pasa
def test_investigacion_valida() -> None:
    assert validar_investigacion(investigacion("politica_envios.pdf", "politica_descuentos.pdf")) == ""


# Test 2: vacía, sin fragmentos y con la fuente caída se distinguen (el reintento sabe qué hacer en cada caso)
def test_investigacion_vacia_o_caida() -> None:
    assert "no entregó" in validar_investigacion({})
    sin_datos = {"consulta": "algo", "fragmentos": [], "resumen": "nada", "fuente_caida": False}
    assert "no recuperó ningún fragmento" in validar_investigacion(sin_datos)
    caida = sin_datos | {"fuente_caida": True}
    assert "falla técnica" in validar_investigacion(caida)


# Test 3: un fragmento que cita un PDF que no existe en el corpus se rechaza
def test_fuente_inexistente() -> None:
    falsa = investigacion("politica_envios.pdf")
    falsa["fragmentos"][0]["fuente"] = "politica_fantasma.pdf"
    assert "no existe en el corpus" in validar_investigacion(falsa)


# Test 4: un fragmento cuyo texto fue alterado no coincide con su id (hash) y se rechaza
def test_fragmento_alterado() -> None:
    alterada = investigacion("politica_envios.pdf")
    alterada["fragmentos"][0]["texto"] += " El envío siempre es gratis."
    assert "fue alterado" in validar_investigacion(alterada)


# Test 5: una categoría que no corresponde al PDF se rechaza
def test_categoria_incoherente() -> None:
    mezclada = investigacion("politica_envios.pdf")
    mezclada["fragmentos"][0]["categoria"] = "vacaciones"
    assert "no existe en el corpus" in validar_investigacion(mezclada)


# ---------- Validación del análisis ----------


# Test 6: un pedido bien calculado, encadenado y respaldado por las políticas recuperadas pasa
def test_analisis_valido() -> None:
    assert validar_analisis(analisis_pedido(), investigacion("politica_descuentos.pdf", "politica_envios.pdf")) == ""


# Test 7: un análisis vacío o sin cálculos se rechaza con un motivo que guía al reintento
def test_analisis_vacio() -> None:
    inv = investigacion("politica_descuentos.pdf")
    assert "no entregó" in validar_analisis({}, inv)
    assert "no ejecutó ningún cálculo" in validar_analisis({"calculos": [], "conclusiones": "nada"}, inv)


# Test 8: el árbitro RECALCULA: un resultado alterado (aunque parezca razonable) no coincide y se rechaza
def test_resultado_alterado() -> None:
    adulterado = copy.deepcopy(analisis_pedido())
    adulterado["calculos"][2]["resultado"]["costo_envio"] = 2500.0
    motivo = validar_analisis(adulterado, investigacion("politica_descuentos.pdf", "politica_envios.pdf"))
    assert "no coincide con el recalculado" in motivo


# Test 9: no se calcula con una política que el investigador no recuperó (grounding)
def test_calculo_sin_respaldo() -> None:
    motivo = validar_analisis(analisis_pedido(), investigacion("politica_descuentos.pdf"))
    assert "politica_envios.pdf" in motivo and "no recuperó" in motivo


# Test 10: el total tiene que usar el descuento que calculó calcular_descuento
def test_descuento_incoherente() -> None:
    analisis = analisis_pedido()
    analisis["calculos"][1] = calculo("calcular_total_pedido", precio_unitario=1200, cantidad=60, descuento_total_pct=10)
    analisis["calculos"][2] = calculo("calcular_envio", monto_compra=analisis["calculos"][1]["resultado"]["total_productos"])
    assert "usó un descuento de 10.0%" in validar_analisis(analisis, investigacion("politica_descuentos.pdf", "politica_envios.pdf"))


# Test 11: el envío se decide sobre el monto YA con descuento (calcularlo sobre el subtotal se detecta)
def test_envio_sobre_subtotal() -> None:
    analisis = analisis_pedido()
    analisis["calculos"][2] = calculo("calcular_envio", monto_compra=72000)
    assert "sobre el monto ya con descuento" in validar_analisis(analisis, investigacion("politica_descuentos.pdf", "politica_envios.pdf"))


# Test 12: entradas inválidas en un cálculo guardado se rechazan sin romper el árbitro
def test_calculo_con_entradas_invalidas() -> None:
    analisis = analisis_pedido()
    analisis["calculos"][2]["entradas"] = {"monto_compra": "mucho"}
    assert "entradas inválidas" in validar_analisis(analisis, investigacion("politica_descuentos.pdf", "politica_envios.pdf"))


# Test 13: el nodo de validación deja el veredicto en el estado y en el registro de contribuciones
async def test_nodo_validacion() -> None:
    estado = estado_inicial("pregunta") | {"ultimo_agente": "investigador", "investigacion": investigacion("politica_envios.pdf")}
    cambios = await nodo_validacion(estado)
    assert cambios["investigacion_valida"] is True and cambios["contribuciones"][0]["valido"] is True
    estado = estado | {"ultimo_agente": "analista", "analisis": {}}
    cambios = await nodo_validacion(estado)
    assert cambios["analisis_valido"] is False and "no entregó" in cambios["error_analisis"]


# ---------- Construcción de resultados desde los mensajes de las herramientas ----------


# Respuesta de la herramienta de búsqueda como la deja un ToolMessage
def mensaje_de_busqueda(fuente: str, score: float = 0.03) -> ToolMessage:
    fr = fragmento(fuente) | {"score": score}
    contenido = {"consulta": "q", "estado": "ok", "modo": "hibrido", "resultados": [fr], "mensaje": ""}
    return ToolMessage(content=json.dumps(contenido), tool_call_id=f"b-{fuente}-{score}", name="buscar_politicas")


# Test 14: la investigación sale SOLO de lo que devolvieron las herramientas; lo inválido se descarta
def test_construir_investigacion() -> None:
    mensajes = [mensaje_de_busqueda("politica_envios.pdf"), ToolMessage(content="esto no es JSON", tool_call_id="x", name="buscar_politicas"), AIMessage(content="Listo")]
    resultado = construir_investigacion("buscar envíos", mensajes)
    assert [f["fuente"] for f in resultado["fragmentos"]] == ["politica_envios.pdf"] and resultado["resumen"] == "Listo" and resultado["fuente_caida"] is False


# Test 15: un refinamiento COMPLETA la investigación previa y, si repite un fragmento, conserva el de mayor puntaje
def test_refinamiento_completa_lo_previo() -> None:
    previos = [fragmento("politica_envios.pdf")]
    mensajes = [mensaje_de_busqueda("politica_descuentos.pdf"), mensaje_de_busqueda("politica_envios.pdf", score=0.5), AIMessage(content="ok")]
    resultado = construir_investigacion("completar", mensajes, previos)
    assert {f["fuente"] for f in resultado["fragmentos"]} == {"politica_envios.pdf", "politica_descuentos.pdf"}
    assert resultado["fragmentos"][0]["score"] == 0.5


# Test 16: una herramienta que falló por la fuente deja la marca fuente_caida (distinta de "no hay datos")
def test_construir_investigacion_fuente_caida() -> None:
    caida = ToolMessage(content="FUENTE_CAIDA: no respondió", tool_call_id="c", name="buscar_politicas", status="error")
    resultado = construir_investigacion("buscar", [caida, AIMessage(content="no pude")])
    assert resultado["fragmentos"] == [] and resultado["fuente_caida"] is True


# Test 17: el análisis sale de los resultados de las herramientas; los argumentos se normalizan y las llamadas fallidas se ignoran
def test_construir_analisis() -> None:
    pedido = AIMessage(content="", tool_calls=[
        {"name": "calcular_descuento", "args": {"cantidad": 60, "es_mayorista": True}, "id": "d", "type": "tool_call"},
        {"name": "calcular_envio", "args": {"monto_compra": -5}, "id": "e", "type": "tool_call"},
    ])
    descuento = ToolMessage(content=json.dumps(recalcular("calcular_descuento", {"cantidad": 60, "es_mayorista": True})), tool_call_id="d", name="calcular_descuento")
    fallido = ToolMessage(content="argumentos inválidos", tool_call_id="e", name="calcular_envio", status="error")
    resultado = construir_analisis([pedido, descuento, fallido, AIMessage(content="13% de descuento")])
    assert [c["herramienta"] for c in resultado["calculos"]] == ["calcular_descuento"]
    assert resultado["calculos"][0]["entradas"] == {"cantidad": 60, "es_mayorista": True}
    assert resultado["calculos"][0]["politica"] == "politica_descuentos.pdf" and resultado["conclusiones"] == "13% de descuento"


# Test 18: la instrucción del analista incluye la orden y los fragmentos recuperados (y nada más)
def test_instruccion_del_analista() -> None:
    texto = armar_instruccion_analista("Calculá el envío", investigacion("politica_envios.pdf"))
    assert texto.startswith("Calculá el envío") and "politica_envios.pdf" in texto


# ---------- Herramientas con contratos Pydantic ----------


# Test 19: buscar_politicas devuelve un JSON validado con la procedencia de cada fragmento
async def test_herramienta_de_busqueda() -> None:
    herramienta = crear_herramienta_busqueda(crear_recuperador_de_prueba(), 3)
    salida = await herramienta.ainvoke({"consulta": "costo de envío menor a 50.000", "categoria": "envios"})
    datos = json.loads(salida)
    assert datos["estado"] == "ok" and datos["modo"] == "hibrido" and 1 <= len(datos["resultados"]) <= 3
    assert all(r["fuente"] == "politica_envios.pdf" and r["fragment_id"] for r in datos["resultados"])


# Test 20: sin resultados NO es un error; una falla de la fuente SÍ lo es (ToolException FUENTE_CAIDA)
async def test_busqueda_sin_resultados_y_con_falla() -> None:
    vacia = await crear_herramienta_busqueda(RecuperadorVacio(), 3).ainvoke({"consulta": "teletrabajo"})
    assert json.loads(vacia)["estado"] == "sin_resultados"
    caida = crear_herramienta_busqueda(RecuperadorIntermitente(crear_recuperador_de_prueba(), fallas=1), 3)
    with pytest.raises(ToolException, match="FUENTE_CAIDA"):
        await caida.ainvoke({"consulta": "envío"})


# Test 21: la herramienta rechaza entradas inválidas (categoría inexistente, consulta vacía, más de 8 resultados)
@pytest.mark.parametrize("argumentos", [{"consulta": "envío", "categoria": "sueldos"}, {"consulta": ""}, {"consulta": "envío", "max_resultados": 50}])
async def test_busqueda_rechaza_entradas_invalidas(argumentos: dict[str, Any]) -> None:
    herramienta = crear_herramienta_busqueda(crear_recuperador_de_prueba(), 3)
    with pytest.raises((ValidationError, ToolException)):
        await herramienta.ainvoke(argumentos)


# Test 22: el analista tiene una herramienta por cálculo; devuelve JSON validado y rechaza entradas inválidas
async def test_herramientas_del_analista() -> None:
    por_nombre = {h.name: h for h in HERRAMIENTAS_ANALISTA}
    assert set(por_nombre) == set(CALCULOS)
    salida = json.loads(await por_nombre["calcular_envio"].ainvoke({"monto_compra": 62640}))
    assert salida == {"politica": "politica_envios.pdf", "envio_gratis": True, "costo_envio": 0.0, "total_con_envio": 62640.0}
    with pytest.raises((ValidationError, ToolException)):
        await por_nombre["calcular_envio"].ainvoke({"monto_compra": -1})
    with pytest.raises((ValidationError, ToolException)):
        await por_nombre["calcular_dias_vacaciones"].ainvoke({"antiguedad_anios": "mucha"})


# ---------- Guardarraíles del supervisor ----------


# Estado de ejemplo del supervisor con los campos que se indiquen
def estado_supervisor(**cambios: Any) -> dict[str, Any]:
    return dict(estado_inicial("Compro 60 unidades de $1.200, ¿cuánto pago?")) | cambios


# Test 23: no se llama a un especialista que agotó sus intentos (se redirige a cerrar)
def test_regla_intentos_agotados() -> None:
    estado = estado_supervisor(intentos_investigador=2, investigacion_valida=False)
    final, nota = aplicar_reglas(Decision(siguiente="investigador", instruccion="x", razon="r"), estado, 2)
    assert final.siguiente == "sintetizador" and "agotaron" in nota


# Test 24: no se analiza sin una investigación validada
def test_regla_no_analizar_sin_investigacion() -> None:
    final, nota = aplicar_reglas(Decision(siguiente="analista", instruccion="", razon="r"), estado_supervisor(), 2)
    assert final.siguiente == "investigador" and "sin una investigación validada" in nota
    assert "Recuperá los fragmentos" in final.instruccion


# Test 25: no se cierra sin investigación validada, ni con un análisis rechazado que todavía se puede corregir
def test_reglas_de_cierre() -> None:
    final, _ = aplicar_reglas(Decision(siguiente="sintetizador", instruccion="", razon="r"), estado_supervisor(), 2)
    assert final.siguiente == "investigador"
    rechazado = estado_supervisor(investigacion_valida=True, intentos_investigador=1, intentos_analista=1, analisis_valido=False, error_analisis="no coincide")
    final, nota = aplicar_reglas(Decision(siguiente="sintetizador", instruccion="", razon="r"), rechazado, 2)
    assert final.siguiente == "analista" and "no coincide" in final.instruccion and nota
    # Con los intentos del analista agotados sí se puede cerrar
    agotado = rechazado | {"intentos_analista": 2}
    assert aplicar_reglas(Decision(siguiente="sintetizador", instruccion="", razon="r"), agotado, 2)[0].siguiente == "sintetizador"


# Test 26: una consulta pura (investigación válida y sin cálculos pedidos) se puede cerrar sin analista
def test_consulta_pura_cierra_sin_analista() -> None:
    estado = estado_supervisor(investigacion_valida=True, intentos_investigador=1)
    final, nota = aplicar_reglas(Decision(siguiente="sintetizador", instruccion="", razon="r"), estado, 2)
    assert final.siguiente == "sintetizador" and nota == "" and final.instruccion == ""


# Test 27: la decisión por defecto (si el LLM falla): investigar, analizar si la pregunta trae cifras, o cerrar
def test_decision_por_defecto() -> None:
    assert decision_por_defecto(estado_supervisor(), 2).siguiente == "investigador"
    con_investigacion = estado_supervisor(investigacion_valida=True, intentos_investigador=1)
    assert decision_por_defecto(con_investigacion, 2).siguiente == "analista"
    sin_cifras = con_investigacion | {"messages": estado_inicial("¿Cuál es el plazo de devolución?")["messages"]}
    assert decision_por_defecto(sin_cifras, 2).siguiente == "sintetizador"


# Test 28: el resumen que ve el supervisor muestra las políticas cubiertas y el estado de cada especialista
def test_resumen_del_estado() -> None:
    estado = estado_supervisor(investigacion_valida=True, intentos_investigador=1, investigacion=investigacion("politica_envios.pdf"))
    resumen = resumir_estado(estado, 2, 8, 2)
    assert "VALIDADA (1 fragmentos de: politica_envios.pdf)" in resumen and "Análisis: NO realizado" in resumen
    assert "NO realizada" in resumir_estado(estado_supervisor(), 1, 8, 2)


# ---------- Errores ----------


# Respuesta HTTP falsa para construir las excepciones reales de los SDK (Any: los SDK tipan su propio httpx interno)
def respuesta_http(codigo: int) -> Any:
    return httpx.Response(codigo, request=httpx.Request("POST", "https://api.ejemplo.test"))


# Pedido HTTP falso (mismo motivo que arriba)
def pedido_http() -> Any:
    return httpx.Request("POST", "https://x.test")


# Test 29: los errores de los proveedores se traducen a mensajes claros y se reconocen los de credenciales
def test_traduccion_de_errores() -> None:
    clave_openai = openai.AuthenticationError("bad key", response=respuesta_http(401), body=None)
    clave_anthropic = anthropic.AuthenticationError("bad key", response=respuesta_http(401), body=None)
    for error in (clave_openai, clave_anthropic):
        assert "API key del proveedor de LLM es inválida" in traducir_error(error)
        assert es_error_de_credenciales(error)
    assert "cuota" in traducir_error(openai.RateLimitError("limit", response=respuesta_http(429), body=None))
    assert "conectar" in traducir_error(anthropic.APIConnectionError(request=pedido_http()))
    assert "Pinecone" in traducir_error(PineconeApiException(status=401, reason="Unauthorized"))
    assert traducir_error(ValueError("algo")) == "ValueError: algo"
    assert not es_error_de_credenciales(ValueError("algo"))


# Test 30: el error de credenciales se reconoce aunque venga envuelto como causa de otro
def test_credenciales_en_la_cadena_de_causas() -> None:
    interno = openai.AuthenticationError("bad key", response=respuesta_http(401), body=None)
    try:
        try:
            raise interno
        except Exception as causa:
            raise RuntimeError("falló la síntesis") from causa
    except RuntimeError as envuelto:
        assert es_error_de_credenciales(envuelto) and "API key" in traducir_error(envuelto)
