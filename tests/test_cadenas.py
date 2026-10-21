# Tests de la cadena de síntesis (LCEL con salida estructurada y reintentos) y de las funciones del sintetizador que arman y componen la respuesta

# Importa asyncio para anular las esperas entre reintentos
import asyncio
# Importa Any para tipar
from typing import Any

# Importa pytest y ValidationError
import pytest
from pydantic import ValidationError

# Importa lo que se prueba
from app.agents.synthesis_agent import (
    armar_datos,
    armar_limitaciones,
    componer_respuesta,
    fuentes_disponibles,
    pregunta_original,
    respuesta_de_respaldo,
)
# Importa la cadena de síntesis y su contrato de salida
from app.cadenas import MontoNoRespaldado, RespuestaFinal, crear_cadena_sintesis, sintetizar
# Importa el corpus real (para armar fragmentos auténticos)
from app.rag.corpus import cargar_chunks
# Importa el estado inicial del orquestador
from app.state import estado_inicial
# Importa los dobles de prueba
from tests.fakes import LLMGuionado, error_de_validacion


# Respuesta correcta de ejemplo
def respuesta_ok() -> RespuestaFinal:
    return RespuestaFinal(respuesta="El envío es gratis desde $50.000.", fuentes_citadas=["politica_envios.pdf"], respondida=True)


# Error con status_code 401, como el que levantan los SDK de OpenAI y Anthropic ante una API key inválida
class ErrorDeClave(Exception):
    status_code = 401


# Datos validados de ejemplo: un fragmento de política que menciona el umbral de $50.000 (respuesta_ok lo cita)
DATOS_ENVIO: dict[str, Any] = {"pregunta": "¿envío?", "fragmentos": [{"fuente": "politica_envios.pdf", "texto": "El envío es gratis en compras desde $50.000."}], "calculos": []}


# Respuesta con un importe que NO está en los datos (un LLM que calcula por su cuenta)
def respuesta_inventada() -> RespuestaFinal:
    return RespuestaFinal(respuesta="El envío cuesta $3.700 y es gratis desde $50.000.", fuentes_citadas=["politica_envios.pdf"], respondida=True)


# Las esperas entre reintentos no corren en los tests (si no, cada reintento tardaría 1 o 2 segundos)
@pytest.fixture(autouse=True)
def sin_esperas(monkeypatch: pytest.MonkeyPatch) -> None:
    real = asyncio.sleep

    # Espera nula: cede el control al event loop pero no duerme
    async def rapido(segundos: float, *args: Any, **kwargs: Any) -> None:
        await real(0)

    monkeypatch.setattr(asyncio, "sleep", rapido)


# Test 1: la cadena devuelve un RespuestaFinal validado y el prompt lleva los datos y el rol de sistema
async def test_la_cadena_devuelve_respuesta_validada() -> None:
    llm = LLMGuionado(respuestas=[respuesta_ok()])
    resultado = await sintetizar(crear_cadena_sintesis(llm), DATOS_ENVIO)
    assert isinstance(resultado, RespuestaFinal) and resultado.respondida
    mensajes = llm.recibidos[0]
    assert "Sintetizador" in str(mensajes[0].content) and "¿envío?" in str(mensajes[1].content)


# Test 2: si la salida viene mal formada (ValidationError), la cadena reintenta y se recupera
async def test_reintenta_ante_salida_mal_formada() -> None:
    llm = LLMGuionado(respuestas=[error_de_validacion(), error_de_validacion(), respuesta_ok()])
    resultado = await sintetizar(crear_cadena_sintesis(llm), DATOS_ENVIO)
    assert resultado.respondida and len(llm.recibidos) == 3


# Test 2b: cada reintento deja un log con el intento y el motivo (criterio de la profesora en la Pre-entrega 2)
async def test_cada_reintento_deja_un_log(caplog: pytest.LogCaptureFixture) -> None:
    llm = LLMGuionado(respuestas=[error_de_validacion(), error_de_validacion(), respuesta_ok()])
    with caplog.at_level("WARNING", logger="app.cadenas"):
        await sintetizar(crear_cadena_sintesis(llm), DATOS_ENVIO)
    avisos = [r.getMessage() for r in caplog.records]
    assert len(avisos) == 2 and "intento 1 de 3" in avisos[0] and "intento 2 de 3" in avisos[1] and "ValidationError" in avisos[0]


# Test 2c: si la respuesta trae un importe que no está en los datos se rechaza, se reintenta avisándole cuál sobra y se acepta la versión corregida
async def test_rechaza_importes_inventados_y_reintenta() -> None:
    llm = LLMGuionado(respuestas=[respuesta_inventada(), respuesta_ok()])
    resultado = await sintetizar(crear_cadena_sintesis(llm), DATOS_ENVIO)
    assert "$3.700" not in resultado.respuesta and len(llm.recibidos) == 2
    # El reintento le dice al LLM qué importe sobraba
    assert "3700" in str(llm.recibidos[1][1].content)


# Test 2d: si el LLM insiste en inventar importes, la síntesis falla (el nodo usa entonces la respuesta de respaldo con los cálculos reales)
async def test_importes_inventados_en_todos_los_intentos() -> None:
    llm = LLMGuionado(respuestas=[respuesta_inventada()] * 3)
    with pytest.raises(MontoNoRespaldado):
        await sintetizar(crear_cadena_sintesis(llm), DATOS_ENVIO)
    assert len(llm.recibidos) == 3


# Test 3: tras 3 intentos fallidos el error se propaga (el sintetizador usa entonces la respuesta de respaldo)
async def test_falla_definitiva_tras_tres_intentos() -> None:
    llm = LLMGuionado(respuestas=[error_de_validacion()] * 3)
    with pytest.raises(ValidationError):
        await sintetizar(crear_cadena_sintesis(llm), {"pregunta": "x"})
    assert len(llm.recibidos) == 3


# Test 4: una API key inválida NO se reintenta (reintentar no la arregla)
async def test_no_reintenta_credenciales() -> None:
    llm = LLMGuionado(respuestas=[ErrorDeClave(), respuesta_ok()])
    with pytest.raises(ErrorDeClave):
        await sintetizar(crear_cadena_sintesis(llm), {"pregunta": "x"})
    assert len(llm.recibidos) == 1


# Test 5: el contrato rechaza una respuesta vacía o demasiado corta
def test_contrato_de_respuesta() -> None:
    with pytest.raises(ValidationError):
        RespuestaFinal(respuesta="ok", fuentes_citadas=[], respondida=True)
    with pytest.raises(ValidationError):
        RespuestaFinal.model_validate({"respuesta": "Una respuesta larga", "fuentes_citadas": "no es lista", "respondida": True})


# Estado de ejemplo con una investigación válida (fragmentos reales del corpus) y, opcionalmente, un análisis
def estado_con(analisis_valido: bool = False, intentos_analista: int = 0) -> dict[str, Any]:
    chunk = next(c for c in cargar_chunks("data", 700, 100) if c.fuente == "politica_envios.pdf")
    estado: dict[str, Any] = dict(estado_inicial("¿Cuánto cuesta el envío?"))
    estado.update(
        investigacion_valida=True, intentos_investigador=1, intentos_analista=intentos_analista, analisis_valido=analisis_valido,
        investigacion={"fragmentos": [{"fuente": chunk.fuente, "pagina": chunk.pagina, "seccion": chunk.seccion, "texto": chunk.texto}]},
    )
    if analisis_valido:
        estado["analisis"] = {"calculos": [{"herramienta": "calcular_envio", "entradas": {"monto_compra": 100.0}, "resultado": {"costo_envio": 2500.0}}]}
    return estado


# Test 6: la pregunta original es el primer mensaje del usuario
def test_pregunta_original() -> None:
    assert pregunta_original(estado_con()) == "¿Cuánto cuesta el envío?"
    assert pregunta_original({}) == ""


# Test 7: limitaciones: una consulta pura (analista nunca llamado) NO tiene limitaciones; un análisis rechazado o una investigación inválida sí
def test_limitaciones() -> None:
    assert armar_limitaciones(estado_con()) == []
    rechazado = estado_con(intentos_analista=1) | {"error_analisis": "el cálculo no coincide"}
    assert armar_limitaciones(rechazado) == ["No se pudo validar el análisis: el cálculo no coincide."]
    sin_investigar = estado_con() | {"investigacion_valida": False, "error_investigacion": "no hubo fragmentos"}
    assert "No se pudo validar la investigación: no hubo fragmentos." in armar_limitaciones(sin_investigar)


# Test 8: los datos para el LLM llevan solo lo validado; lo rechazado no se le muestra
def test_datos_para_el_llm() -> None:
    datos = armar_datos(estado_con(analisis_valido=True, intentos_analista=1), [])
    assert datos["fragmentos"][0]["fuente"] == "politica_envios.pdf" and datos["calculos"][0]["herramienta"] == "calcular_envio"
    rechazado = estado_con(intentos_analista=1) | {"error_analisis": "mal"}
    assert "calculos" not in armar_datos(rechazado, ["limitación"]) and armar_datos(rechazado, ["limitación"])["limitaciones"] == ["limitación"]
    assert "fragmentos" not in armar_datos(estado_con() | {"investigacion_valida": False}, [])


# Test 9: componer_respuesta descarta fuentes inventadas, completa las que faltan y agrega la línea "Fuentes:"
def test_componer_respuesta() -> None:
    disponibles = ["politica_envios.pdf"]
    inventada = RespuestaFinal(respuesta="Envío gratis desde $50.000.", fuentes_citadas=["politica_fantasma.pdf"], respondida=True)
    texto, fuentes = componer_respuesta(inventada, disponibles)
    assert fuentes == disponibles and texto.endswith("Fuentes: politica_envios.pdf")
    # Si no pudo responder, no cita fuentes aunque las haya
    no_responde = RespuestaFinal(respuesta="Las políticas no cubren ese tema.", fuentes_citadas=[], respondida=False)
    assert componer_respuesta(no_responde, disponibles) == ("Las políticas no cubren ese tema.", [])
    # Aunque el LLM nombre una fuente real, si no respondió no se cita: lo recuperado no sustenta ninguna respuesta
    cita_sin_responder = RespuestaFinal(respuesta="Las políticas no cubren ese tema.", fuentes_citadas=["politica_envios.pdf"], respondida=False)
    assert componer_respuesta(cita_sin_responder, disponibles) == ("Las políticas no cubren ese tema.", [])
    # Sin repetidos
    repetida = RespuestaFinal(respuesta="Envío gratis desde $50.000.", fuentes_citadas=["politica_envios.pdf"] * 2, respondida=True)
    assert componer_respuesta(repetida, disponibles)[1] == disponibles


# Test 10: las fuentes disponibles son las de la investigación validada, sin repetir; sin investigación válida no hay fuentes
def test_fuentes_disponibles() -> None:
    assert fuentes_disponibles(estado_con()) == ["politica_envios.pdf"]
    assert fuentes_disponibles(estado_con() | {"investigacion_valida": False}) == []


# Test 11: la respuesta de respaldo (sin LLM) informa los cálculos o las fuentes, y siempre las limitaciones
def test_respuesta_de_respaldo() -> None:
    con_calculos = respuesta_de_respaldo(estado_con(analisis_valido=True, intentos_analista=1), [])
    assert "calcular_envio" in con_calculos and "2500.0" in con_calculos
    assert "politica_envios.pdf" in respuesta_de_respaldo(estado_con(), [])
    assert respuesta_de_respaldo({}, []) == "No se pudo generar una respuesta."
    assert "Falta algo." in respuesta_de_respaldo(estado_con(), ["Falta algo."])
