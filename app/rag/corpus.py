# Módulo del corpus: lee los PDF de las políticas y los parte en fragmentos (chunks) con su metadata (fuente, página, categoría, sección)
# El troceado es DETERMINISTA: el mismo PDF siempre produce los mismos fragmentos con los mismos ids.
# Por eso la ingesta (que sube los vectores a Pinecone) y el recuperador (que arma el BM25 local) ven exactamente el mismo corpus

# Importa asyncio para correr la lectura de PDF (bloqueante) en un hilo y no frenar el event loop
import asyncio
# Importa hashlib para calcular el id de cada fragmento a partir de su contenido
import hashlib
# Importa logging para dejar registro de lo que se carga
import logging
# Importa re para detectar los títulos de sección ("1. Costo de envío")
import re
# Importa dataclass para definir cada fragmento como un objeto inmutable
from dataclasses import dataclass
# Importa Path para armar las rutas a los PDF
from pathlib import Path
# Importa Literal para tipar las categorías
from typing import Literal

# Importa PdfReader para extraer el texto de los PDF
from pypdf import PdfReader

# Crea el logger del módulo
logger = logging.getLogger(__name__)

# Categorías de política (son el filtro por metadata que acepta el recuperador)
Categoria = Literal["descuentos", "devoluciones", "envios", "vacaciones"]

# Para cada categoría, el PDF que la contiene (es el corpus completo del sistema)
CATEGORIAS: dict[str, str] = {
    "descuentos": "politica_descuentos.pdf",
    "devoluciones": "politica_devoluciones.pdf",
    "envios": "politica_envios.pdf",
    "vacaciones": "politica_vacaciones.pdf",
}

# Un título de sección empieza con número y punto ("2. Costo de envío") y ocupa toda la línea
PATRON_SECCION = re.compile(r"^\d+\.\s+\S.*$", re.MULTILINE)


# Un fragmento del corpus: la unidad que se indexa, se recupera y se cita
@dataclass(frozen=True)
class Chunk:
    # Identificador estable: hash de fuente + página + texto
    chunk_id: str
    # PDF de origen
    fuente: str
    # Página (empieza en 1)
    pagina: int
    # Categoría de la política (filtro por metadata)
    categoria: str
    # Título de la sección a la que pertenece
    seccion: str
    # Texto del fragmento (incluye el título del documento y de la sección para que se entienda solo)
    texto: str


# Calcula el id de un fragmento. Es público porque el árbitro (validation.py) lo usa para comprobar que un hallazgo no fue alterado
def id_fragmento(fuente: str, pagina: int, texto: str) -> str:
    return hashlib.sha256(f"{fuente}|{pagina}|{texto}".encode()).hexdigest()[:16]


# Parte un texto largo en ventanas de hasta "tamano" caracteres con "solape" de superposición, cortando en espacios para no partir palabras
def partir_texto(texto: str, tamano: int, solape: int) -> list[str]:
    # Si entra en una sola ventana, no hay nada que partir
    if len(texto) <= tamano:
        return [texto]
    ventanas: list[str] = []
    inicio = 0
    while inicio < len(texto):
        fin = min(inicio + tamano, len(texto))
        # Si no es la última ventana, retrocede hasta el último espacio para no cortar una palabra
        if fin < len(texto):
            corte = texto.rfind(" ", inicio + solape + 1, fin)
            fin = corte if corte != -1 else fin
        ventanas.append(texto[inicio:fin].strip())
        # La última ventana termina el recorrido
        if fin >= len(texto):
            break
        # La próxima arranca "solape" caracteres antes del corte (siempre avanza porque solape < tamano)
        inicio = fin - solape
        # Si cayó en medio de una palabra, avanza hasta el comienzo de la siguiente para no empezar la ventana con una palabra cortada
        if texto[inicio - 1] != " ":
            espacio = texto.find(" ", inicio)
            inicio = espacio + 1 if espacio != -1 else inicio
    return ventanas


# Parte el texto de UNA página en secciones: devuelve (título de sección, cuerpo) y deja el título del documento aparte
def separar_secciones(texto_pagina: str) -> tuple[str, list[tuple[str, str]]]:
    lineas = [linea.strip() for linea in texto_pagina.splitlines() if linea.strip()]
    # La primera línea es el título del documento (los PDF de políticas lo traen arriba)
    titulo_documento = lineas[0] if lineas else ""
    contenido = "\n".join(lineas[1:])
    # Busca dónde empieza cada título de sección
    inicios = [coincidencia.start() for coincidencia in PATRON_SECCION.finditer(contenido)]
    # Sin secciones numeradas, toda la página es una sola sección
    if not inicios:
        return titulo_documento, [("", contenido)] if contenido else []
    secciones: list[tuple[str, str]] = []
    # El texto anterior a la primera sección (si hay) se conserva como una sección sin título
    if inicios[0] > 0 and contenido[: inicios[0]].strip():
        secciones.append(("", contenido[: inicios[0]].strip()))
    for posicion, inicio in enumerate(inicios):
        fin = inicios[posicion + 1] if posicion + 1 < len(inicios) else len(contenido)
        bloque = contenido[inicio:fin].strip()
        titulo, _, cuerpo = bloque.partition("\n")
        secciones.append((titulo.strip(), cuerpo.strip()))
    return titulo_documento, secciones


# Lee un PDF y devuelve el texto de cada página (bloqueante: se llama desde un hilo)
def leer_paginas(ruta: Path) -> list[str]:
    return [(pagina.extract_text() or "") for pagina in PdfReader(str(ruta)).pages]


# Carga todo el corpus: lee los 4 PDF y devuelve sus fragmentos en un orden estable (bloqueante)
def cargar_chunks(datos_dir: str | Path, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    carpeta = Path(datos_dir)
    chunks: list[Chunk] = []
    for categoria, fuente in CATEGORIAS.items():
        ruta = carpeta / fuente
        if not ruta.is_file():
            raise FileNotFoundError(f"No se encontró {ruta}. Los PDF de las políticas tienen que estar en la carpeta '{carpeta}'.")
        for numero_pagina, texto_pagina in enumerate(leer_paginas(ruta), start=1):
            titulo_documento, secciones = separar_secciones(texto_pagina)
            for titulo_seccion, cuerpo in secciones:
                # Cada fragmento arranca con el documento y la sección: así se entiende aunque se lea suelto
                encabezado = " — ".join(parte for parte in (titulo_documento, titulo_seccion) if parte)
                for ventana in partir_texto(cuerpo, chunk_size, chunk_overlap):
                    texto = f"{encabezado}\n{ventana}" if encabezado else ventana
                    chunks.append(
                        Chunk(
                            chunk_id=id_fragmento(fuente, numero_pagina, texto),
                            fuente=fuente,
                            pagina=numero_pagina,
                            categoria=categoria,
                            seccion=titulo_seccion,
                            texto=texto,
                        )
                    )
    logger.info("Corpus cargado: %s fragmentos de %s documentos", len(chunks), len(CATEGORIAS))
    return chunks


# Versión asíncrona: la lectura de PDF es trabajo de CPU y disco, así que corre en un hilo aparte
async def cargar_chunks_async(datos_dir: str | Path, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    return await asyncio.to_thread(cargar_chunks, datos_dir, chunk_size, chunk_overlap)
