# Tests del recuperador híbrido (BM25 + vectores + fusión RRF) sobre el corpus REAL; solo el embedder y el almacén vectorial son falsos

# Importa json para leer el golden set
import json
# Importa Path para ubicarlo
from pathlib import Path

# Importa pytest para comprobar excepciones
import pytest

# Importa lo que se prueba
from app.rag.recuperador import fusion_rrf, tokenizar
# Importa los dobles de prueba
from tests.fakes import AlmacenEnMemoria, EmbedderFalso, chunks_de_prueba, crear_recuperador_de_prueba
from app.rag.recuperador import RecuperadorHibrido

# Preguntas con su documento esperado (el golden set de la Pre-entrega 4)
GOLDEN = json.loads(Path("data/golden_set.json").read_text(encoding="utf-8"))


# Test 1: la tokenización ignora tildes, mayúsculas y palabras vacías, y reduce a la raíz ("devolución" ~ "devolver")
def test_tokenizar() -> None:
    assert tokenizar("¿Cuál es el COSTO de Envío?") == tokenizar("costo envio")
    assert tokenizar("devoluciones") == tokenizar("devolución")
    assert tokenizar("el de la") == []


# Test 2: RRF suma 1/(k+posición) por cada ranking: lo que aparece en los dos sube
def test_fusion_rrf() -> None:
    puntajes = fusion_rrf([["a", "b", "c"], ["b", "d"]], 60)
    assert puntajes["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert puntajes["a"] == pytest.approx(1 / 61)
    assert max(puntajes, key=lambda clave: puntajes[clave]) == "b"
    assert fusion_rrf([], 60) == {}


# Test 3: el golden set: para cada pregunta, el documento esperado está entre los 3 primeros (hit@3) y la mayoría es el primero
async def test_golden_set_hit_at_3() -> None:
    recuperador = crear_recuperador_de_prueba()
    aciertos_en_1 = 0
    for caso in GOLDEN:
        resultado = await recuperador.buscar(caso["pregunta"], 3)
        fuentes = [f.fuente for f in resultado.fragmentos]
        assert caso["documento_id_esperado"] in fuentes, caso["pregunta"]
        aciertos_en_1 += fuentes[0] == caso["documento_id_esperado"]
    assert aciertos_en_1 >= len(GOLDEN) - 1


# Test 4: los fragmentos devueltos llevan su procedencia completa y vienen ordenados de mayor a menor puntaje
async def test_procedencia_y_orden() -> None:
    resultado = await crear_recuperador_de_prueba().buscar("costo de envío menor a 50.000", 4)
    assert resultado.modo == "hibrido" and len(resultado.fragmentos) == 4
    puntajes = [f.score for f in resultado.fragmentos]
    assert puntajes == sorted(puntajes, reverse=True)
    assert all(f.fuente and f.pagina >= 1 and f.categoria and f.fragment_id for f in resultado.fragmentos)


# Test 5: el filtro por categoría restringe las dos ramas (BM25 y vectores)
async def test_filtro_por_categoria() -> None:
    recuperador = crear_recuperador_de_prueba()
    for categoria in ("descuentos", "devoluciones", "envios", "vacaciones"):
        resultado = await recuperador.buscar("plazo costo días porcentaje", 5, categoria)
        assert resultado.fragmentos and {f.categoria for f in resultado.fragmentos} == {categoria}
    # El filtro llegó al almacén vectorial
    almacen = recuperador.almacen
    assert isinstance(almacen, AlmacenEnMemoria) and almacen.consultas[-1] == (recuperador.candidatos, "vacaciones")


# Test 6: si Pinecone falla, degrada a solo BM25 y lo informa (no se cae)
async def test_degradacion_a_bm25() -> None:
    recuperador = crear_recuperador_de_prueba(falla_vectorial=ConnectionError("Pinecone caído"))
    resultado = await recuperador.buscar("devolver un producto plazo", 3)
    assert resultado.modo == "solo_bm25" and resultado.fragmentos[0].fuente == "politica_devoluciones.pdf"


# Test 7: si Pinecone falla y BM25 tampoco encuentra nada, el error se propaga (no hay con qué responder)
async def test_sin_nada_propaga_el_error() -> None:
    recuperador = crear_recuperador_de_prueba(falla_vectorial=ConnectionError("Pinecone caído"))
    with pytest.raises(ConnectionError):
        await recuperador.buscar("zzzz qqqq", 3)


# Test 8: un id que Pinecone devuelve pero ya no está en el corpus local (índice viejo) se ignora
async def test_ids_viejos_se_ignoran() -> None:
    chunks = chunks_de_prueba()
    embedder = EmbedderFalso()
    almacen = AlmacenEnMemoria()
    await almacen.guardar(chunks, await embedder.embeber_documentos([c.texto for c in chunks]))
    # Un vector "huérfano" idéntico a la consulta: sería el primero si no se ignorara
    almacen.datos["id-viejo-que-ya-no-existe"] = (embedder.vector("costo de envío"), "envios")
    recuperador = RecuperadorHibrido(chunks, embedder, almacen)
    resultado = await recuperador.buscar("costo de envío", 5)
    assert "id-viejo-que-ya-no-existe" not in [f.fragment_id for f in resultado.fragmentos]
    assert resultado.fragmentos


# Test 9: el embedding de la consulta se pide una sola vez por búsqueda
async def test_un_embedding_por_busqueda() -> None:
    recuperador = crear_recuperador_de_prueba()
    await recuperador.buscar("vacaciones antigüedad", 3)
    assert isinstance(recuperador.embedder, EmbedderFalso) and recuperador.embedder.llamadas_consulta == 1
