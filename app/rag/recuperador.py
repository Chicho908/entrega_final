# Módulo del recuperador híbrido: combina una búsqueda por palabras (BM25, local) con una búsqueda semántica (vectores en Pinecone)
# y fusiona los dos rankings con RRF (Reciprocal Rank Fusion). Es el "RAG avanzado" de la Pre-entrega 4, ahora 100% asíncrono
#
#   consulta ─┬→ BM25 (hilo aparte) ───────────────┐
#             └→ embedding → Pinecone (async) ─────┴→ RRF → top-k fragmentos con fuente, página y categoría
#
# Si la rama vectorial falla (Pinecone caído), el recuperador degrada a BM25 y lo informa en el campo "modo" en vez de caerse

# Importa asyncio para correr las dos ramas en paralelo y BM25 en un hilo
import asyncio
# Importa logging para avisar las degradaciones
import logging
# Importa re y unicodedata para normalizar el texto (minúsculas, sin tildes) antes de tokenizar
import re
import unicodedata
# Importa Literal y Protocol para tipar el resultado y el contrato
from typing import Literal, Protocol

# Importa BM25Okapi: el algoritmo de ranking por palabras
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
# Importa el stemmer en español (reduce "devolución" y "devolver" a una raíz parecida)
import snowballstemmer  # type: ignore[import-untyped]
# Importa BaseModel para el resultado validado
from pydantic import BaseModel

# Importa el contrato del embedder y del almacén
from app.rag.almacen import AlmacenVectorial
from app.rag.corpus import Chunk
from app.rag.embeddings import Embedder
# Importa el contrato del fragmento de política
from app.state import FragmentoPolitica

# Crea el logger del módulo
logger = logging.getLogger(__name__)

# Palabras muy comunes que no ayudan a distinguir un documento de otro
PALABRAS_VACIAS = frozenset(
    "a al algo ante como con cual cuales cuando cuanto cuantos cuanta cuantas de del donde el ella en es esta este esto "
    "hay la las le les lo los me mi mis muy no o para pero por que se si sin sobre su sus te tiene tienen un una uno unos unas y ya".split()
)

# Stemmer de español (se crea una vez; no tiene estado)
STEMMER = snowballstemmer.stemmer("spanish")


# Resultado de una búsqueda: los fragmentos y cómo se obtuvieron
class ResultadoBusqueda(BaseModel):
    # Fragmentos ordenados de más a menos relevante
    fragmentos: list[FragmentoPolitica]
    # "hibrido" = BM25 + vectores; "solo_bm25" = Pinecone no respondió y se usó solo la búsqueda por palabras
    modo: Literal["hibrido", "solo_bm25"]


# Contrato de un recuperador (los tests de agentes usan uno falso; en producción es el híbrido)
class Recuperador(Protocol):
    # Busca los k fragmentos más relevantes para la consulta, opcionalmente dentro de una categoría
    async def buscar(self, consulta: str, k: int, categoria: str | None = None) -> ResultadoBusqueda: ...


# Quita tildes y pasa a minúsculas ("Envíos" → "envios")
def normalizar(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFD", texto.lower())
    return "".join(caracter for caracter in descompuesto if unicodedata.category(caracter) != "Mn")


# Convierte un texto en la lista de raíces que usa BM25 (sin palabras vacías)
def tokenizar(texto: str) -> list[str]:
    palabras = re.findall(r"[a-z0-9]+", normalizar(texto))
    return [STEMMER.stemWord(palabra) for palabra in palabras if palabra not in PALABRAS_VACIAS]


# Fusiona varios rankings con RRF: cada documento suma 1 / (k + posición) por cada ranking en el que aparece
def fusion_rrf(rankings: list[list[str]], k: int) -> dict[str, float]:
    puntajes: dict[str, float] = {}
    for ranking in rankings:
        for posicion, chunk_id in enumerate(ranking, start=1):
            puntajes[chunk_id] = puntajes.get(chunk_id, 0.0) + 1.0 / (k + posicion)
    return puntajes


# Recuperador híbrido real
class RecuperadorHibrido:
    # Arma el índice BM25 en memoria con el corpus (son pocos fragmentos: se construye en milisegundos al arrancar)
    def __init__(self, chunks: list[Chunk], embedder: Embedder, almacen: AlmacenVectorial, rrf_k: int = 60, candidatos: int = 10) -> None:
        self.chunks = chunks
        self.por_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.embedder = embedder
        self.almacen = almacen
        self.rrf_k = rrf_k
        self.candidatos = candidatos
        self.bm25 = BM25Okapi([tokenizar(chunk.texto) for chunk in chunks])

    # Ranking por palabras (bloqueante: CPU pura, por eso se llama desde un hilo). Solo incluye fragmentos con al menos una palabra en común
    def _ranking_bm25(self, consulta: str, categoria: str | None) -> list[str]:
        puntajes = self.bm25.get_scores(tokenizar(consulta))
        candidatos = [
            (puntaje, chunk.chunk_id)
            for puntaje, chunk in zip(puntajes, self.chunks, strict=True)
            if puntaje > 0 and (categoria is None or chunk.categoria == categoria)
        ]
        candidatos.sort(key=lambda par: (-par[0], par[1]))
        return [chunk_id for _, chunk_id in candidatos[: self.candidatos]]

    # Ranking semántico: embedding de la consulta + búsqueda en Pinecone (con el filtro por categoría)
    async def _ranking_vectorial(self, consulta: str, categoria: str | None) -> list[str]:
        vector = await self.embedder.embeber_consulta(consulta)
        coincidencias = await self.almacen.buscar(vector, self.candidatos, categoria)
        return [chunk_id for chunk_id, _ in coincidencias]

    # Busca los k fragmentos más relevantes combinando las dos señales
    async def buscar(self, consulta: str, k: int, categoria: str | None = None) -> ResultadoBusqueda:
        # Las dos ramas corren en paralelo; return_exceptions permite degradar si solo falla la vectorial
        bm25, vectorial = await asyncio.gather(
            asyncio.to_thread(self._ranking_bm25, consulta, categoria),
            self._ranking_vectorial(consulta, categoria),
            return_exceptions=True,
        )
        # BM25 es local: si falla es un bug, no una caída de infraestructura
        if isinstance(bm25, BaseException):
            raise bm25
        rankings = [bm25]
        modo: Literal["hibrido", "solo_bm25"] = "hibrido"
        if isinstance(vectorial, BaseException):
            # Una cancelación (apagado del worker) no se disimula
            if isinstance(vectorial, asyncio.CancelledError):
                raise vectorial
            # Si tampoco hubo resultados por palabras, no queda nada con qué responder: el error se propaga
            if not bm25:
                raise vectorial
            logger.warning("La búsqueda vectorial falló (%s): se usa solo BM25", type(vectorial).__name__)
            modo = "solo_bm25"
        else:
            rankings.append(vectorial)
        puntajes = fusion_rrf(rankings, self.rrf_k)
        ordenados = sorted(puntajes.items(), key=lambda par: (-par[1], par[0]))
        fragmentos: list[FragmentoPolitica] = []
        for chunk_id, puntaje in ordenados:
            chunk = self.por_id.get(chunk_id)
            # Un vector que Pinecone devuelve pero ya no está en el corpus local (índice viejo) se ignora
            if chunk is None:
                logger.warning("Pinecone devolvió el id %s, que no está en el corpus local; se ignora", chunk_id)
                continue
            fragmentos.append(
                FragmentoPolitica(
                    fragment_id=chunk.chunk_id,
                    fuente=chunk.fuente,
                    pagina=chunk.pagina,
                    categoria=chunk.categoria,
                    seccion=chunk.seccion,
                    texto=chunk.texto,
                    score=round(puntaje, 6),
                )
            )
            if len(fragmentos) == k:
                break
        return ResultadoBusqueda(fragmentos=fragmentos, modo=modo)
