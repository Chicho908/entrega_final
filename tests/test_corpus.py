# Tests del corpus (app/rag/corpus.py): lectura de los PDF y troceado determinista con metadata

# Importa Path para comprobar rutas
from pathlib import Path

# Importa pytest para comprobar excepciones
import pytest

# Importa lo que se prueba
from app.rag.corpus import CATEGORIAS, cargar_chunks, cargar_chunks_async, id_fragmento, partir_texto, separar_secciones


# Test 1: los 4 PDF producen 13 fragmentos, todos con fuente, página, categoría y sección, y ids únicos
def test_corpus_real_completo() -> None:
    chunks = cargar_chunks("data", 700, 100)
    assert len(chunks) == 13
    assert {c.fuente for c in chunks} == set(CATEGORIAS.values())
    assert {c.categoria for c in chunks} == set(CATEGORIAS)
    assert len({c.chunk_id for c in chunks}) == 13
    assert all(c.pagina >= 1 and c.seccion and c.texto for c in chunks)
    # Cada categoría apunta a su propio PDF
    assert all(CATEGORIAS[c.categoria] == c.fuente for c in chunks)


# Test 2: el troceado es determinista: dos cargas dan exactamente los mismos fragmentos con los mismos ids (la ingesta y el BM25 ven lo mismo)
def test_troceado_determinista() -> None:
    assert cargar_chunks("data", 700, 100) == cargar_chunks("data", 700, 100)


# Test 3: el id depende del contenido: cambiar una coma cambia el id (así el árbitro detecta fragmentos alterados)
def test_id_depende_del_contenido() -> None:
    assert id_fragmento("a.pdf", 1, "texto") == id_fragmento("a.pdf", 1, "texto")
    assert id_fragmento("a.pdf", 1, "texto") != id_fragmento("a.pdf", 1, "texto.")
    assert id_fragmento("a.pdf", 1, "texto") != id_fragmento("a.pdf", 2, "texto")


# Test 4: cada fragmento arranca con el título del documento y de la sección (se entiende aunque se lea suelto)
def test_fragmentos_llevan_encabezado() -> None:
    envio = next(c for c in cargar_chunks("data", 700, 100) if c.fuente == "politica_envios.pdf")
    primera_linea = envio.texto.splitlines()[0]
    assert "—" in primera_linea and envio.seccion in primera_linea


# Test 5: partir_texto: un texto corto queda entero; uno largo se parte con solape, sin cortar palabras y cubriendo todo el texto
def test_partir_texto() -> None:
    assert partir_texto("hola mundo", 100, 10) == ["hola mundo"]
    texto = " ".join(f"palabra{i}" for i in range(200))
    ventanas = partir_texto(texto, 120, 30)
    assert len(ventanas) > 1 and all(len(v) <= 120 for v in ventanas)
    # Ninguna ventana corta una palabra por la mitad
    palabras_validas = set(texto.split())
    assert all(p in palabras_validas for v in ventanas for p in v.split())
    # Hay solape entre ventanas consecutivas y la última palabra del texto llega a la última ventana
    assert set(ventanas[0].split()) & set(ventanas[1].split())
    assert ventanas[-1].endswith("palabra199")


# Test 6: separar_secciones detecta los títulos numerados y deja el título del documento aparte
def test_separar_secciones() -> None:
    titulo, secciones = separar_secciones("Politica de envios\n1. Costo\nSe cobra 2500.\n2. Plazo\nTarda 3 días.")
    assert titulo == "Politica de envios"
    assert secciones == [("1. Costo", "Se cobra 2500."), ("2. Plazo", "Tarda 3 días.")]
    # Sin secciones numeradas toda la página es una sola sección sin título
    assert separar_secciones("Titulo\nsolo texto")[1] == [("", "solo texto")]
    assert separar_secciones("") == ("", [])


# Test 7: si falta un PDF, el error dice cuál y dónde buscarlo
def test_pdf_faltante(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="politica_descuentos.pdf"):
        cargar_chunks(tmp_path, 700, 100)


# Test 8: la versión asíncrona (lectura en un hilo) devuelve lo mismo que la síncrona
async def test_carga_asincrona() -> None:
    assert await cargar_chunks_async("data", 700, 100) == cargar_chunks("data", 700, 100)
