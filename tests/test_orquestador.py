# Tests del orquestador: se ejecutan SIN API key ni Pinecone usando LLM falsos que siguen un guion
# Las herramientas, el RAG (BM25 + RRF), la validación, los guardarraíles y el grafo son los REALES; solo cambian los "cerebros" externos
# Cubren: delegación completa, ruta sin analista, estado, aislamiento de contexto, supervisor infinito, guardarraíles, validación y degradación honesta

# Importa asyncio para correr las funciones asíncronas dentro de tests normales
import asyncio
# Importa Any para tipar
from typing import Any

# Importa pytest para los tests y para esperar excepciones
import pytest
# Importa los tipos de mensajes
from langchain_core.messages import AIMessage, HumanMessage

# Importa la configuración, que también se prueba
from app import config
# Importa el grupo de LLM y el armado del grafo
from app.cadenas import RespuestaFinal
from app.graph import LLMs, crear_grafo
# Importa el script que regenera el diagrama del grafo
from scripts.generar_diagrama import generar_mermaid
# Importa el contrato de decisión y el estado inicial
from app.state import Decision, estado_inicial
# Importa los guardarraíles del supervisor
from app.supervisor import aplicar_reglas
# Importa los dobles de prueba compartidos
from tests.fakes import (
    PREGUNTA, TOTAL_ESPERADO, LLMGuionado, RecuperadorIntermitente, armar_orquestador, crear_recuperador_de_prueba, decidir, guion_analista,
    guion_investigador, guion_supervisor_feliz, guion_sintetizador, pedir, settings_de_prueba,
)

# ---------- Ayudas ----------


# Corre el orquestador y devuelve (estado final, lista de nodos en el orden en que se ejecutaron)
def correr(app: Any, pregunta: str = PREGUNTA) -> tuple[dict[str, Any], list[str]]:
    # Recorre el flujo y junta los nodos ejecutados y el estado final
    async def _correr() -> tuple[dict[str, Any], list[str]]:
        nodos: list[str] = []
        final: dict[str, Any] = {}
        async for modo, contenido in app.astream(estado_inicial(pregunta), config={"recursion_limit": 60}, stream_mode=["updates", "values"]):
            if modo == "updates":
                nodos.extend(contenido.keys())
            else:
                final = contenido
        return final, nodos

    return asyncio.run(_correr())


# Texto del primer mensaje humano que recibió un LLM falso en una de sus llamadas
def humano_de(llm: LLMGuionado, llamada: int) -> str:
    for mensaje in llm.recibidos[llamada]:
        if isinstance(mensaje, HumanMessage):
            return str(mensaje.content)
    return ""


# Búsqueda del investigador que no devuelve nada útil (para provocar rechazos de la validación)
BUSQUEDA_FALLIDA = [pedir(("buscar_politicas", {"consulta": "descuento mayorista"}), prefijo="x"), AIMessage(content="La fuente no respondió.")]

# ---------- Flujo principal ----------


# Test 1: delegación completa (investigador → analista → síntesis) con validación entre cada paso
def test_delegacion_completa() -> None:
    app, _ = armar_orquestador(guion_supervisor_feliz(), guion_investigador(), guion_analista(), guion_sintetizador())
    estado, nodos = correr(app)
    # El orden de ejecución es exactamente el de la consigna, con la validación después de cada especialista
    assert nodos == ["supervisor", "investigador", "validacion", "supervisor", "analista", "validacion", "supervisor", "sintetizador"]
    # Ambos resultados pasaron la validación
    assert estado["investigacion_valida"] and estado["analisis_valido"]
    # Los fragmentos son los reales del corpus (con su procedencia) y los números los calcularon las herramientas
    assert {f["fuente"] for f in estado["investigacion"]["fragmentos"]} >= {"politica_descuentos.pdf", "politica_envios.pdf"}
    totales = [c["resultado"]["total_productos"] for c in estado["analisis"]["calculos"] if c["herramienta"] == "calcular_total_pedido"]
    assert totales == [TOTAL_ESPERADO]
    # La respuesta final llegó al estado y al historial de mensajes, con sus fuentes
    assert estado["respuesta_final"].startswith("El cliente paga $62.640")
    assert estado["respuesta_final"].endswith("Fuentes: politica_descuentos.pdf, politica_envios.pdf")
    assert estado["fuentes"] == ["politica_descuentos.pdf", "politica_envios.pdf"]
    assert isinstance(estado["messages"][-1], AIMessage) and estado["messages"][-1].content == estado["respuesta_final"]


# Test 2: consulta pura: el supervisor pasa del investigador directo al sintetizador, sin analista
def test_consulta_pura_sin_analista() -> None:
    investigador = [
        pedir(("buscar_politicas", {"consulta": "plazo en días corridos para devolver un producto", "categoria": "devoluciones"}), prefijo="i"),
        AIMessage(content="La política de devoluciones fija el plazo."),
    ]
    sintesis = [RespuestaFinal(respuesta="El cliente tiene 10 días corridos desde que recibe el producto.", fuentes_citadas=["politica_devoluciones.pdf"], respondida=True)]
    supervisor = [decidir("investigador", "Recuperá el plazo de devolución."), decidir("sintetizador", "", "es una consulta: no pide cálculos")]
    app, _ = armar_orquestador(supervisor, investigador, [], sintesis)
    estado, nodos = correr(app, "¿Cuántos días corridos tiene un cliente para devolver un producto?")
    assert nodos == ["supervisor", "investigador", "validacion", "supervisor", "sintetizador"]
    assert estado["intentos_analista"] == 0 and not estado["analisis_valido"]
    assert estado["fuentes"] == ["politica_devoluciones.pdf"]
    # Que el análisis no se haya hecho NO es una limitación: la pregunta no lo necesitaba
    assert "No se pudo validar el análisis" not in str(estado["respuesta_final"])


# Test 3: el estado rastrea qué agente contribuyó con qué (y la validación deja su veredicto)
def test_estado_rastrea_contribuciones() -> None:
    app, _ = armar_orquestador(guion_supervisor_feliz(), guion_investigador(), guion_analista(), guion_sintetizador())
    estado, _ = correr(app)
    agentes = [c["agente"] for c in estado["contribuciones"]]
    assert agentes == ["investigador", "validacion", "analista", "validacion", "sintetizador"]
    # El aporte del investigador incluye las herramientas que usó (procedencia)
    herramientas = [p["herramienta"] for p in estado["contribuciones"][0]["pasos"] if p["tipo"] == "decide_herramienta"]
    assert herramientas == ["buscar_politicas", "buscar_politicas"]
    # Cada decisión del supervisor queda auditada
    assert [d["final"] for d in estado["decisiones"]] == ["investigador", "analista", "sintetizador"]


# Test 4: contexto aislado: cada especialista ve SOLO su instrucción, no la pregunta original ni el historial
def test_contexto_aislado() -> None:
    app, llms = armar_orquestador(guion_supervisor_feliz(), guion_investigador(), guion_analista(), guion_sintetizador())
    correr(app)
    # El investigador recibió: 1 mensaje de sistema (su rol) y 1 humano con la instrucción del supervisor
    primera = llms.investigador.recibidos[0]
    assert [type(m).__name__ for m in primera] == ["SystemMessage", "HumanMessage"]
    assert humano_de(llms.investigador, 0).startswith("Recuperá las políticas de descuento por cantidad")
    # El analista recibió su instrucción + los fragmentos de política, pero NO la pregunta original del usuario
    texto_analista = humano_de(llms.analista, 0)
    assert "Calculá el descuento de 60 unidades" in texto_analista and "politica_descuentos.pdf" in texto_analista
    assert PREGUNTA not in texto_analista
    # El supervisor recibe un resumen del estado (con las políticas recuperadas), no el historial de mensajes de los especialistas
    resumen = humano_de(llms.supervisor, 1)
    assert "Investigación: VALIDADA" in resumen and "politica_envios.pdf" in resumen and "tool" not in resumen.lower()


# Test 5: el sintetizador recibe datos estructurados (fragmentos y cálculos reales), no el resumen en texto libre del investigador
def test_sintesis_usa_datos_estructurados() -> None:
    app, llms = armar_orquestador(guion_supervisor_feliz(), guion_investigador(), guion_analista(), guion_sintetizador())
    correr(app)
    datos = humano_de(llms.sintetizador, 0)
    assert '"fragmentos"' in datos and '"calculos"' in datos and '"total_productos": 62640.0' in datos
    assert "Recuperé la política de descuentos" not in datos


# ---------- Supervisor infinito y guardarraíles ----------


# Test 6: un supervisor que nunca termina se corta por el tope de pasos (no llega al límite de recursión)
def test_supervisor_infinito_se_corta_por_tope_de_pasos() -> None:
    # Investigador que nunca encuentra nada + muchos refinamientos permitidos: solo el tope de pasos puede frenarlo
    configuracion = settings_de_prueba(max_pasos_supervisor=4, max_refinamientos=100)
    sintesis = [RespuestaFinal(respuesta="No hay datos suficientes para responder.", fuentes_citadas=[], respondida=False)]
    app, llms = armar_orquestador(
        [decidir("investigador")], [AIMessage(content="No encontré nada.")], [AIMessage(content="-")], sintesis,
        configuracion=configuracion, repetir_supervisor=True,
    )
    llms.investigador.repetir_ultima = True
    estado, nodos = correr(app)
    # El supervisor decidió exactamente 4 veces y la última fue el cierre forzado
    assert estado["pasos_supervisor"] == 4
    assert estado["decisiones"][-1]["final"] == "sintetizador"
    assert "tope" in estado["decisiones"][-1]["guardarrail"]
    # El investigador corrió 3 veces (pasos 1 a 3) y el flujo terminó con una respuesta
    assert estado["intentos_investigador"] == 3
    assert nodos[-1] == "sintetizador" and estado["respuesta_final"]


# Test 7: el supervisor pide analizar sin investigación → el guardarraíl lo redirige al investigador
def test_guardarrail_no_analizar_sin_investigacion() -> None:
    supervisor = [decidir("analista"), decidir("analista"), decidir("sintetizador")]
    app, _ = armar_orquestador(supervisor, guion_investigador(), guion_analista(), guion_sintetizador())
    estado, nodos = correr(app)
    assert estado["decisiones"][0]["propuesta"] == "analista" and estado["decisiones"][0]["final"] == "investigador"
    assert "sin una investigación validada" in estado["decisiones"][0]["guardarrail"]
    assert nodos[1] == "investigador"
    assert estado["analisis_valido"]


# Test 8: el supervisor intenta cerrar sin haber investigado → la rúbrica lo obliga a empezar por el investigador
def test_guardarrail_no_cerrar_sin_investigacion() -> None:
    supervisor = [decidir("sintetizador"), decidir("analista"), decidir("sintetizador")]
    app, _ = armar_orquestador(supervisor, guion_investigador(), guion_analista(), guion_sintetizador())
    estado, _ = correr(app)
    assert estado["decisiones"][0]["final"] == "investigador"
    assert "rúbrica no se cumple" in estado["decisiones"][0]["guardarrail"]
    assert estado["investigacion_valida"] and estado["analisis_valido"]


# Test 9: si el LLM del supervisor falla, decide la regla por defecto (la pregunta trae cifras → analiza) y el flujo se completa igual
def test_supervisor_con_llm_caido_usa_regla_por_defecto() -> None:
    app, _ = armar_orquestador([RuntimeError("caído")], guion_investigador(), guion_analista(), guion_sintetizador(), repetir_supervisor=True)
    estado, nodos = correr(app)
    assert [d["propuesta"] for d in estado["decisiones"]] == ["(falló el LLM)"] * 3
    assert nodos.count("investigador") == 1 and nodos.count("analista") == 1 and nodos[-1] == "sintetizador"
    assert estado["investigacion_valida"] and estado["analisis_valido"]


# ---------- Validación, refinamiento y conflictos ----------


# Test 10: si la búsqueda falla (Pinecone caído), la investigación es rechazada y el supervisor reintenta informando el error específico
def test_refinamiento_de_investigacion_con_error_especifico() -> None:
    recuperador = RecuperadorIntermitente(crear_recuperador_de_prueba(), fallas=1)
    investigador = BUSQUEDA_FALLIDA + guion_investigador()
    supervisor = [decidir("investigador", "Buscá las políticas."), decidir("investigador", "Probá de nuevo."), decidir("analista"), decidir("sintetizador")]
    app, llms = armar_orquestador(supervisor, investigador, guion_analista(), guion_sintetizador(), recuperador=recuperador)
    estado, _ = correr(app)
    # El primer intento fue rechazado con un motivo concreto: falla técnica, no falta de datos
    assert estado["contribuciones"][1]["agente"] == "validacion" and not estado["contribuciones"][1]["valido"]
    assert "falla técnica" in estado["contribuciones"][1]["resumen"]
    # El segundo intento recibió ese motivo en su instrucción y se corrigió
    segunda_instruccion = humano_de(llms.investigador, 2)
    assert "Corregí este problema" in segunda_instruccion and "no respondió" in segunda_instruccion
    assert estado["intentos_investigador"] == 2 and estado["investigacion_valida"] and estado["analisis_valido"]


# Test 11: conflicto entre cálculos: el analista calcula el envío sobre el subtotal (sin descuento) → el árbitro lo rechaza y se corrige
def test_conflicto_numerico_el_arbitro_rechaza() -> None:
    analista = guion_analista(monto_envio=72000.0) + guion_analista()
    supervisor = [decidir("investigador"), decidir("analista"), decidir("analista"), decidir("sintetizador")]
    app, llms = armar_orquestador(supervisor, guion_investigador(), analista, guion_sintetizador())
    estado, _ = correr(app)
    veredictos = [c for c in estado["contribuciones"] if c["agente"] == "validacion" and c["sobre"] == "analista"]
    assert not veredictos[0]["valido"] and "el envío se decide sobre el monto ya con descuento" in veredictos[0]["resumen"]
    assert veredictos[1]["valido"]
    assert estado["intentos_analista"] == 2
    # El reintento del analista recibió el error específico
    assert "el envío se decide sobre el monto ya con descuento" in humano_de(llms.analista, 4)


# Test 12: cálculo sin respaldo: el analista usa una política que el investigador no recuperó → se rechaza, se investiga y la evidencia se COMPLETA
def test_calculo_sin_respaldo_obliga_a_investigar_la_politica() -> None:
    solo_descuentos = [pedir(("buscar_politicas", {"consulta": "descuento mayorista", "categoria": "descuentos"}), prefijo="a"), AIMessage(content="Solo descuentos.")]
    solo_envios = [pedir(("buscar_politicas", {"consulta": "costo de envío", "categoria": "envios"}), prefijo="b"), AIMessage(content="Envíos.")]
    supervisor = [decidir("investigador"), decidir("analista"), decidir("investigador", "Recuperá la política de envíos."), decidir("analista"), decidir("sintetizador")]
    app, _ = armar_orquestador(supervisor, solo_descuentos + solo_envios, guion_analista() * 2, guion_sintetizador())
    estado, _ = correr(app)
    veredictos = [c for c in estado["contribuciones"] if c["agente"] == "validacion" and c["sobre"] == "analista"]
    assert not veredictos[0]["valido"] and "no recuperó esa política" in veredictos[0]["resumen"]
    # La segunda investigación sumó la política de envíos a la de descuentos (no la reemplazó)
    assert {f["fuente"] for f in estado["investigacion"]["fragmentos"]} == {"politica_descuentos.pdf", "politica_envios.pdf"}
    assert estado["analisis_valido"] and estado["intentos_investigador"] == 2 and estado["intentos_analista"] == 2


# Test 13: intentos agotados → degradación honesta: se cierra sin respuesta inventada y la síntesis informa la limitación
def test_degradacion_honesta_al_agotar_intentos() -> None:
    recuperador = RecuperadorIntermitente(crear_recuperador_de_prueba(), fallas=100)
    sintesis = [RespuestaFinal(respuesta="No pude consultar las políticas, por eso no puedo responder.", fuentes_citadas=[], respondida=False)]
    supervisor = [decidir("investigador")] * 3
    app, llms = armar_orquestador(supervisor, BUSQUEDA_FALLIDA * 2, guion_analista(), sintesis, recuperador=recuperador)
    estado, nodos = correr(app)
    assert estado["intentos_investigador"] == 2 and estado["intentos_analista"] == 0 and "analista" not in nodos
    assert not estado["investigacion_valida"]
    assert "agotaron los intentos" in estado["decisiones"][-1]["guardarrail"]
    # El sintetizador recibió la limitación y ningún dato sin validar
    datos = humano_de(llms.sintetizador, 0)
    assert "No se pudo validar la investigación" in datos and '"fragmentos"' not in datos
    assert estado["fuentes"] == []


# Test 14: un especialista que falla (excepción del LLM) no rompe el grafo: queda registrado y la validación lo rechaza
def test_falla_de_especialista_no_rompe_el_grafo() -> None:
    configuracion = settings_de_prueba(max_refinamientos=0)
    sintesis = [RespuestaFinal(respuesta="Cierre explicando la limitación.", fuentes_citadas=[], respondida=False)]
    app, _ = armar_orquestador([decidir("investigador")] * 2, [RuntimeError("API caída")], guion_analista(), sintesis, configuracion=configuracion)
    estado, nodos = correr(app)
    assert "RuntimeError" in estado["contribuciones"][0]["error"]
    assert estado["error_investigacion"] == "el investigador no entregó ningún resultado"
    assert nodos[-1] == "sintetizador"


# Test 15: un especialista atrapado en su propio bucle ReAct se corta por su límite de iteraciones
def test_limite_de_iteraciones_del_especialista() -> None:
    configuracion = settings_de_prueba(max_refinamientos=0, limite_iteraciones_agente=2)
    bucle = LLMGuionado(respuestas=[pedir(("buscar_politicas", {"consulta": "envío"}))], repetir_ultima=True)
    app, llms = armar_orquestador([decidir("investigador")] * 2, [], guion_analista(), guion_sintetizador(), configuracion=configuracion)
    # Reemplaza el LLM del investigador por uno que pide herramientas para siempre
    llms2 = LLMs(supervisor=llms.supervisor, investigador=bucle, analista=llms.analista, sintetizador=llms.sintetizador)
    estado, _ = correr(crear_grafo(llms2, configuracion, crear_recuperador_de_prueba()).compile())
    assert "límite de 2 iteraciones" in estado["contribuciones"][0]["error"]


# ---------- Reglas del supervisor (unitarias) ----------


# Test 16: tabla de decisiones del guardarraíl
@pytest.mark.parametrize(
    ("propuesta", "estado", "esperado"),
    [
        # Sin investigación: nadie puede analizar ni cerrar
        ("investigador", {}, "investigador"),
        ("analista", {}, "investigador"),
        ("sintetizador", {}, "investigador"),
        # Con investigación válida y sin análisis intentado: se puede cerrar directo (consulta pura) o analizar
        ("sintetizador", {"investigacion_valida": True, "intentos_investigador": 1}, "sintetizador"),
        ("analista", {"investigacion_valida": True, "intentos_investigador": 1}, "analista"),
        # Análisis rechazado con intentos disponibles: no se cierra
        ("sintetizador", {"investigacion_valida": True, "intentos_investigador": 1, "intentos_analista": 1}, "analista"),
        ("sintetizador", {"investigacion_valida": True, "analisis_valido": True, "intentos_investigador": 1, "intentos_analista": 1}, "sintetizador"),
        # Intentos agotados
        ("investigador", {"intentos_investigador": 2}, "sintetizador"),
        ("investigador", {"investigacion_valida": True, "intentos_investigador": 2, "intentos_analista": 1}, "analista"),
        ("analista", {"investigacion_valida": True, "intentos_investigador": 1, "intentos_analista": 2}, "sintetizador"),
    ],
)
def test_tabla_de_guardarrailes(propuesta: str, estado: dict[str, Any], esperado: str) -> None:
    decision, _ = aplicar_reglas(Decision(siguiente=propuesta, instruccion="x", razon="r"), estado, max_intentos=2)  # type: ignore[arg-type]
    assert decision.siguiente == esperado


# ---------- Configuración y diagrama ----------


# Variables obligatorias de una configuración válida
VARIABLES_OBLIGATORIAS = {
    "LLM_MODEL": "modelo-de-prueba", "PINECONE_API_KEY": "pc-prueba", "PINECONE_INDEX_NAME": "indice-de-prueba",
    "PINECONE_NAMESPACE": "ns-de-prueba", "PINECONE_CLOUD": "nube-de-prueba", "PINECONE_REGION": "region-de-prueba",
    "EMBEDDING_MODEL": "embeddings-de-prueba", "EMBEDDING_DIM": "1024",
}


# Test 17: la configuración valida las variables al arrancar y nada importante tiene valor escrito en el código
def test_configuracion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    for variable in (
        "LLM_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MAX_REFINAMIENTOS", "MAX_PASOS_SUPERVISOR", "RECURSION_LIMIT",
        "CHUNK_SIZE", "CHUNK_OVERLAP", *VARIABLES_OBLIGATORIAS,
    ):
        monkeypatch.delenv(variable, raising=False)
    # Sin API key del LLM corta con un mensaje claro (el proveedor por defecto es anthropic)
    with pytest.raises(config.ConfigError, match="ANTHROPIC_API_KEY"):
        config.cargar_configuracion()
    # El valor de ejemplo del .env.example tampoco sirve
    monkeypatch.setenv("ANTHROPIC_API_KEY", "tu_api_key_aqui")
    with pytest.raises(config.ConfigError):
        config.cargar_configuracion()
    # Con la key real, cada variable obligatoria que falta se reclama por su nombre (modelo, Pinecone, índice, namespace, nube, región, embeddings y su dimensión)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-prueba")
    for variable, valor in VARIABLES_OBLIGATORIAS.items():
        with pytest.raises(config.ConfigError, match=variable):
            config.cargar_configuracion()
        monkeypatch.setenv(variable, valor)
    # Con todo completo usa los valores por defecto y no muestra las keys en el repr
    ajustes = config.cargar_configuracion()
    assert ajustes.max_refinamientos == 1 and ajustes.max_pasos_supervisor == 8 and ajustes.modelo == "modelo-de-prueba"
    assert ajustes.pinecone_index == "indice-de-prueba" and ajustes.embedding_dim == 1024 and ajustes.top_k == 4
    assert "sk-prueba" not in repr(ajustes) and "pc-prueba" not in repr(ajustes)
    # MAX_REFINAMIENTOS=0 es válido (sin reintentos) y un valor inválido se rechaza
    monkeypatch.setenv("MAX_REFINAMIENTOS", "0")
    assert config.cargar_configuracion().max_refinamientos == 0
    monkeypatch.setenv("MAX_PASOS_SUPERVISOR", "1")
    with pytest.raises(config.ConfigError, match="MAX_PASOS_SUPERVISOR"):
        config.cargar_configuracion()
    # Un RECURSION_LIMIT demasiado bajo para el tope de pasos del supervisor se rechaza al arrancar
    monkeypatch.setenv("MAX_PASOS_SUPERVISOR", "8")
    monkeypatch.setenv("RECURSION_LIMIT", "10")
    with pytest.raises(config.ConfigError, match="RECURSION_LIMIT"):
        config.cargar_configuracion()
    # El solapamiento de los fragmentos no puede ser igual o mayor que su tamaño
    monkeypatch.setenv("RECURSION_LIMIT", "40")
    monkeypatch.setenv("CHUNK_OVERLAP", "700")
    with pytest.raises(config.ConfigError, match="CHUNK_OVERLAP"):
        config.cargar_configuracion()


# Test 17b: el nombre del proyecto de Phoenix no está escrito en el código: con endpoint activo es obligatorio y sin endpoint no hace falta
def test_configuracion_nombre_de_proyecto_de_phoenix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-prueba")
    for variable, valor in VARIABLES_OBLIGATORIAS.items():
        monkeypatch.setenv(variable, valor)
    monkeypatch.delenv("PHOENIX_PROJECT_NAME", raising=False)
    # Sin endpoint la observabilidad está apagada y el nombre queda vacío (no hay ningún valor por defecto en el código)
    monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)
    assert config.cargar_configuracion().phoenix_proyecto == ""
    # Con endpoint pero sin nombre de proyecto corta al arrancar, nombrando la variable que falta
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:6006/v1/traces")
    with pytest.raises(config.ConfigError, match="PHOENIX_PROJECT_NAME"):
        config.cargar_configuracion()
    # Con el nombre definido usa exactamente el que viene del entorno
    monkeypatch.setenv("PHOENIX_PROJECT_NAME", "proyecto-de-prueba")
    assert config.cargar_configuracion().phoenix_proyecto == "proyecto-de-prueba"


# Test 18: el diagrama Mermaid del grafo contiene los nodos y las aristas condicionales del supervisor
def test_diagrama_mermaid() -> None:
    app, _ = armar_orquestador([], [], [], [])
    mermaid = app.get_graph().draw_mermaid()
    for nodo in ("supervisor", "investigador", "analista", "validacion", "sintetizador", "aprobacion", "despacho"):
        assert nodo in mermaid
    assert "supervisor -.-> investigador" in mermaid and "validacion --> supervisor" in mermaid
    # Tramo HITL: solo las tareas críticas pasan por la aprobación humana
    assert "sintetizador -.-> aprobacion" in mermaid and "aprobacion --> despacho" in mermaid


# Test 19: el script que regenera el diagrama funciona sin .env ni red y dibuja los 7 nodos del grafo real
def test_script_del_diagrama() -> None:
    mermaid = generar_mermaid()
    for nodo in ("supervisor", "investigador", "analista", "validacion", "sintetizador", "aprobacion", "despacho"):
        assert nodo in mermaid
