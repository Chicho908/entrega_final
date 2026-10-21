# Test del script de pruebas del sistema (scripts/pruebas_sistema.py): los 5 escenarios corren contra la API real en memoria (grafo real, LLM falsos y Redis falso)
# Sirve para no descubrir un error del script gastando llamadas reales al LLM

# Importa AsyncIterator para tipar los recursos de prueba
from collections.abc import AsyncIterator
# Importa los context managers para armar los recursos de prueba
from contextlib import asynccontextmanager

# Importa httpx para el cliente en memoria
import httpx
# Importa los mensajes del LLM falso
from langchain_core.messages import AIMessage
# Importa el checkpointer en memoria
from langgraph.checkpoint.memory import InMemorySaver
# Importa Redis (solo para tipar)
from redis.asyncio import Redis

# Importa el script a probar
from scripts.pruebas_sistema import ESCENARIOS, ejecutar, sin_separadores
# Importa la respuesta estructurada de la cadena y el almacén de trabajos
from app.cadenas import RespuestaFinal
from app.jobs import JobStore
# Importa la aplicación y el runtime
from app.main import AbridorRuntime, crear_app
from app.runtime import Runtime
# Importa los dobles de prueba
from tests.fakes import (
    DespachadorEspia, armar_orquestador, decidir, guion_analista, guion_investigador, guion_supervisor_feliz, guion_sintetizador, pedir, settings_de_prueba,
)


# Guiones de los 5 escenarios, en el orden en que corren (con un solo worker los trabajos se ejecutan de a uno)
def guiones() -> tuple[list[object], list[object], list[object], list[object]]:
    supervisor = [
        # 1. consulta pura: investiga y cierra
        decidir("investigador", "Recuperá el plazo de devolución."), decidir("sintetizador"),
        # 2. pedido
        *guion_supervisor_feliz(),
        # 3. vacaciones
        decidir("investigador", "Recuperá los días de vacaciones por antigüedad y cómo se solicitan."), decidir("analista", "Calculá los días para 7 años."), decidir("sintetizador"),
        # 4. tarea crítica (mismo camino que el pedido)
        *guion_supervisor_feliz(),
        # 5. fuera de dominio
        decidir("investigador", "Buscá si hay política de teletrabajo."), decidir("sintetizador"),
    ]
    investigador = [
        pedir(("buscar_politicas", {"consulta": "plazo para devolver un producto", "categoria": "devoluciones"}), prefijo="a"),
        AIMessage(content="Recuperé la política de devoluciones."),
        *guion_investigador(),
        pedir(("buscar_politicas", {"consulta": "días de vacaciones según antigüedad", "categoria": "vacaciones"}), ("buscar_politicas", {"consulta": "cómo solicitar las vacaciones", "categoria": "vacaciones"}), prefijo="v"),
        AIMessage(content="Recuperé la política de vacaciones."),
        *guion_investigador(),
        pedir(("buscar_politicas", {"consulta": "teletrabajo días por semana"}), prefijo="t"),
        AIMessage(content="No encontré una política de teletrabajo."),
    ]
    analista = [
        *guion_analista(),
        pedir(("calcular_dias_vacaciones", {"antiguedad_anios": 7}), prefijo="vac"),
        AIMessage(content="Con 7 años corresponden 21 días corridos."),
        *guion_analista(),
    ]
    sintetizador = [
        RespuestaFinal(respuesta="El cliente tiene 10 días corridos desde que recibe el producto.", fuentes_citadas=["politica_devoluciones.pdf"], respondida=True),
        *guion_sintetizador(),
        RespuestaFinal(respuesta="Corresponden 21 días corridos y hay que pedirlos con 15 días de anticipación.", fuentes_citadas=["politica_vacaciones.pdf"], respondida=True),
        *guion_sintetizador(),
        RespuestaFinal(respuesta="Las políticas no tienen información sobre teletrabajo.", fuentes_citadas=[], respondida=False),
    ]
    return supervisor, investigador, analista, sintetizador


# Levanta la API con el grafo real y los guiones de los 5 escenarios; entrega un cliente HTTP en memoria
@asynccontextmanager
async def cliente(redis: Redis, espia: DespachadorEspia) -> AsyncIterator[httpx.AsyncClient]:
    supervisor, investigador, analista, sintetizador = guiones()

    # Abre los recursos de la aplicación con el grafo real y los guiones de los 5 escenarios
    @asynccontextmanager
    async def abrir() -> AsyncIterator[Runtime]:
        grafo, _ = armar_orquestador(supervisor, investigador, analista, sintetizador, checkpointer=InMemorySaver(), despachador=espia)
        yield Runtime(settings_de_prueba(workers_embebidos=1), JobStore(redis), grafo)

    abridor: AbridorRuntime = abrir
    aplicacion = crear_app(abridor)
    async with aplicacion.router.lifespan_context(aplicacion):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=aplicacion), base_url="http://prueba") as http:
            yield http


# Test 1: los 5 escenarios del script pasan de punta a punta, incluida la pausa y la aprobación del escenario crítico
async def test_los_cinco_escenarios_pasan(redis: Redis) -> None:
    espia = DespachadorEspia()
    async with cliente(redis, espia) as http:
        previos, resultados = await ejecutar(http, timeout_s=20, pausa_s=0.02)
    assert previos == []
    assert len(resultados) == len(ESCENARIOS) == 5
    assert [r.problemas for r in resultados] == [[], [], [], [], []]
    # Solo el escenario crítico envió algo, y recién después de la aprobación
    assert [destinatario for destinatario, _, _ in espia.enviados] == ["gerencia@empresa.com"]
    # Cada escenario trae un job_id distinto (la traza de cada uno se busca por ese id)
    assert len({r.job_id for r in resultados}) == 5


# Test 2: si el sistema responde mal, el script lo detecta y lo informa (no da todo por bueno)
async def test_el_script_detecta_respuestas_incorrectas(redis: Redis) -> None:
    espia = DespachadorEspia()
    supervisor, investigador, analista, sintetizador = guiones()
    # El escenario 5 inventa una fuente para una pregunta que las políticas no cubren
    sintetizador[-1] = RespuestaFinal(respuesta="Se puede teletrabajar dos días por semana.", fuentes_citadas=["politica_vacaciones.pdf"], respondida=True)

    # Abre los recursos con el escenario 5 saboteado
    @asynccontextmanager
    async def abrir() -> AsyncIterator[Runtime]:
        grafo, _ = armar_orquestador(supervisor, investigador, analista, sintetizador, checkpointer=InMemorySaver(), despachador=espia)
        yield Runtime(settings_de_prueba(workers_embebidos=1), JobStore(redis), grafo)

    aplicacion = crear_app(abrir)
    async with aplicacion.router.lifespan_context(aplicacion):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=aplicacion), base_url="http://prueba") as http:
            _, resultados = await ejecutar(http, timeout_s=20, pausa_s=0.02)
    assert [r.problemas for r in resultados[:4]] == [[], [], [], []]
    assert "inventada" in resultados[4].problemas[0]


# Test 3: sin_separadores deja las cifras comparables sin importar el formato ($62.640, 62,640 o 62640)
def test_sin_separadores() -> None:
    assert {sin_separadores(t) for t in ("$62.640", "$ 62,640", "$62640")} == {"$62640"}
