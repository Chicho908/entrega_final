# Tests de la integración con Pinecone: el SDK asíncrono OFICIAL habla por HTTP con un servidor falso (tests/pinecone_falso.py)
# Se comprueba lo que un doble de Python no puede: las rutas, los cuerpos y los filtros que arma el SDK, y cómo parsea las respuestas

# Importa asyncio para anular las esperas entre reintentos
import asyncio
# Importa AsyncIterator para tipar la fixture
from collections.abc import AsyncIterator

# Importa pytest y pytest_asyncio para las fixtures
import pytest
import pytest_asyncio
# Importa el cliente asíncrono oficial de Pinecone
from pinecone import PineconeAsyncio
# Importa las excepciones de Pinecone
from pinecone.exceptions import UnauthorizedException

# Importa lo que se prueba
from app.config import Settings
from app.rag.almacen import AlmacenPinecone, ErrorIndice, asegurar_indice, obtener_host
from app.rag.corpus import cargar_chunks
from app.rag.embeddings import EmbedderPinecone
from app.rag.ingesta import ingestar
from app.rag.recuperador import RecuperadorHibrido
# Importa el servidor falso
from tests.fakes import settings_de_prueba
from tests.pinecone_falso import PineconeFalso


# Servidor falso de Pinecone levantado para el test
@pytest_asyncio.fixture
async def servidor() -> AsyncIterator[PineconeFalso]:
    falso = PineconeFalso()
    await falso.iniciar()
    yield falso
    await falso.detener()


# Cliente real del SDK apuntando al servidor falso
@pytest_asyncio.fixture
async def cliente(servidor: PineconeFalso) -> AsyncIterator[PineconeAsyncio]:
    async with PineconeAsyncio(api_key="clave-de-prueba", host=servidor.url) as pc:
        yield pc


# Configuración de prueba con nombres de índice, namespace y dimensión del servidor falso
def ajustes(servidor: PineconeFalso) -> Settings:
    return settings_de_prueba(
        pinecone_api_key="clave-de-prueba", pinecone_index="indice-de-prueba", pinecone_namespace="ns-de-prueba",
        embedding_model="modelo-de-prueba", embedding_dim=servidor.dimension, datos_dir="data",
    )


# Test 1: asegurar_indice crea el índice con la dimensión de la configuración y devuelve su host; la segunda vez no lo vuelve a crear
async def test_asegurar_indice_crea_una_sola_vez(servidor: PineconeFalso, cliente: PineconeAsyncio) -> None:
    host = await asegurar_indice(cliente, "indice-de-prueba", servidor.dimension, "aws", "us-east-1")
    assert host == servidor.url
    creaciones = [p for p in servidor.pedidos if p[0] == "POST" and p[1] == "/indexes"]
    assert len(creaciones) == 1 and creaciones[0][2]["dimension"] == servidor.dimension and creaciones[0][2]["spec"]["serverless"]["cloud"] == "aws"
    await asegurar_indice(cliente, "indice-de-prueba", servidor.dimension, "aws", "us-east-1")
    assert len([p for p in servidor.pedidos if p[0] == "POST" and p[1] == "/indexes"]) == 1


# Test 2: un índice con otra dimensión se rechaza con un mensaje claro (el clásico "mismatch de dimensiones")
async def test_dimension_incorrecta_se_detecta(servidor: PineconeFalso, cliente: PineconeAsyncio) -> None:
    await asegurar_indice(cliente, "indice-de-prueba", servidor.dimension, "aws", "us-east-1")
    with pytest.raises(ErrorIndice, match="dimensión"):
        await asegurar_indice(cliente, "indice-de-prueba", 1024, "aws", "us-east-1")
    with pytest.raises(ErrorIndice, match="dimensión"):
        await obtener_host(cliente, "indice-de-prueba", 1024)


# Test 3: obtener_host falla con un mensaje que explica cómo resolverlo si el índice no existe (la API no crea índices)
async def test_obtener_host_sin_indice(cliente: PineconeAsyncio) -> None:
    with pytest.raises(ErrorIndice, match="python -m app.rag.ingesta"):
        await obtener_host(cliente, "no-existe", 32)


# Test 4: el embedder usa input_type distinto para pasajes y para consultas, y devuelve un vector por texto
async def test_embeddings_pasaje_y_consulta(servidor: PineconeFalso, cliente: PineconeAsyncio) -> None:
    embedder = EmbedderPinecone(cliente, "modelo-de-prueba")
    vectores = await embedder.embeber_documentos(["envío gratis", "plazo de devolución"])
    consulta = await embedder.embeber_consulta("costo de envío")
    assert len(vectores) == 2 and all(len(v) == servidor.dimension for v in vectores) and len(consulta) == servidor.dimension
    pedidos = [p[2] for p in servidor.pedidos if p[1] == "/embed"]
    assert pedidos[0]["parameters"]["input_type"] == "passage" and pedidos[1]["parameters"]["input_type"] == "query"
    assert pedidos[0]["model"] == "modelo-de-prueba" and pedidos[0]["parameters"]["truncate"] == "END"


# Test 5: ingesta de punta a punta con el SDK real: crea el índice, sube los 13 fragmentos con su metadata y es idempotente
async def test_ingesta_completa_e_idempotente(servidor: PineconeFalso, monkeypatch: pytest.MonkeyPatch) -> None:
    # El SDK lee el host del control plane de esta variable cuando no se lo pasan por parámetro
    monkeypatch.setenv("PINECONE_CONTROLLER_HOST", servidor.url)
    configuracion = ajustes(servidor)
    assert await ingestar(configuracion) == 13
    assert await ingestar(configuracion) == 13
    guardados = servidor.vectores[("indice", "ns-de-prueba")]
    assert len(guardados) == 13
    ejemplo = next(iter(guardados.values()))
    assert set(ejemplo["metadata"]) == {"fuente", "pagina", "categoria", "seccion"} and len(ejemplo["values"]) == servidor.dimension
    assert servidor.api_keys == {"clave-de-prueba"}


# Test 6: la búsqueda híbrida completa (embedding → query con filtro por categoría → fusión RRF) contra el SDK real
async def test_busqueda_hibrida_con_el_sdk_real(servidor: PineconeFalso, cliente: PineconeAsyncio) -> None:
    host = await asegurar_indice(cliente, "indice-de-prueba", servidor.dimension, "aws", "us-east-1")
    chunks = cargar_chunks("data", 700, 100)
    embedder = EmbedderPinecone(cliente, "modelo-de-prueba")
    async with cliente.IndexAsyncio(host=host) as indice:
        almacen = AlmacenPinecone(indice, "ns-de-prueba")
        assert await almacen.guardar(chunks, await embedder.embeber_documentos([c.texto for c in chunks])) == 13
        recuperador = RecuperadorHibrido(chunks, embedder, almacen)
        resultado = await recuperador.buscar("cuánto cuesta el envío si compro menos de 50.000", 3)
        assert resultado.modo == "hibrido" and resultado.fragmentos[0].fuente == "politica_envios.pdf"
        # El filtro por categoría viaja a Pinecone como filtro de metadata ($eq)
        solo_vacaciones = await recuperador.buscar("anticipación para pedir", 3, categoria="vacaciones")
        assert {f.categoria for f in solo_vacaciones.fragmentos} == {"vacaciones"}
        consultas = [p[2] for p in servidor.pedidos if p[1] == "/query"]
        assert consultas[-1]["filter"] == {"categoria": {"$eq": "vacaciones"}} and consultas[0].get("filter") is None


# Test 7: una falla transitoria (503) se reintenta y se recupera; una API key inválida (401) NO se reintenta
async def test_reintentos_ante_fallas_transitorias(servidor: PineconeFalso, cliente: PineconeAsyncio, monkeypatch: pytest.MonkeyPatch) -> None:
    # Sin esperas reales entre reintentos
    async def sin_espera(segundos: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", sin_espera)
    embedder = EmbedderPinecone(cliente, "modelo-de-prueba")
    servidor.fallas["/embed"] = [503]
    assert len(await embedder.embeber_consulta("envío")) == servidor.dimension
    assert len([p for p in servidor.pedidos if p[1] == "/embed"]) == 2
    servidor.fallas["/embed"] = [401]
    with pytest.raises(UnauthorizedException):
        await embedder.embeber_consulta("envío")
    assert len([p for p in servidor.pedidos if p[1] == "/embed"]) == 3
