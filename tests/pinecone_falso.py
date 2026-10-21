# Servidor HTTP que imita a Pinecone (control plane, data plane e inference) para probar el SDK asíncrono OFICIAL sin red ni API key
# El SDK real arma los pedidos y parsea las respuestas: así se comprueba la integración a nivel HTTP (rutas, cuerpos, filtros, tipos), no solo contra un doble de Python

# Importa math y zlib para armar embeddings falsos deterministas
import math
import zlib
# Importa Any para tipar
from typing import Any

# Importa los bloques del servidor web de aiohttp (ya es dependencia del SDK asíncrono de Pinecone)
from aiohttp import web
from aiohttp.test_utils import TestServer

# Importa el tokenizador del recuperador (para que los embeddings falsos respeten la similitud por palabras)
from app.rag.recuperador import tokenizar


# Servidor falso de Pinecone con estado en memoria
class PineconeFalso:
    # Crea el estado: índices creados, vectores por (índice, namespace) y registro de los pedidos recibidos
    def __init__(self, dimension: int = 32) -> None:
        self.dimension = dimension
        self.indices: dict[str, dict[str, Any]] = {}
        self.vectores: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        self.pedidos: list[tuple[str, str, dict[str, Any]]] = []
        self.api_keys: set[str] = set()
        # Si se completa, el próximo pedido a esa ruta falla con ese código HTTP (para probar reintentos); se descuenta en cada falla
        self.fallas: dict[str, list[int]] = {}
        self.servidor: TestServer | None = None

    # URL base del servidor (sirve de host de control plane y de data plane)
    @property
    def url(self) -> str:
        assert self.servidor is not None
        return str(self.servidor.make_url("")).rstrip("/")

    # Arranca el servidor en un puerto libre
    async def iniciar(self) -> None:
        aplicacion = web.Application()
        aplicacion.router.add_get("/indexes", self.listar)
        aplicacion.router.add_post("/indexes", self.crear)
        aplicacion.router.add_get("/indexes/{nombre}", self.describir)
        aplicacion.router.add_post("/embed", self.embeber)
        aplicacion.router.add_post("/vectors/upsert", self.upsert)
        aplicacion.router.add_post("/query", self.consultar)
        self.servidor = TestServer(aplicacion)
        await self.servidor.start_server()

    # Apaga el servidor
    async def detener(self) -> None:
        if self.servidor is not None:
            await self.servidor.close()

    # Vector falso de un texto: bolsa de palabras con hashing, normalizado (igual que el EmbedderFalso de los tests)
    def vector(self, texto: str) -> list[float]:
        valores = [0.0] * self.dimension
        for palabra in tokenizar(texto):
            valores[zlib.crc32(palabra.encode()) % self.dimension] += 1.0
        norma = math.sqrt(sum(v * v for v in valores)) or 1.0
        return [v / norma for v in valores]

    # Registra el pedido y simula una falla si hay una programada para la ruta. Devuelve la respuesta de error o None
    async def _entrar(self, pedido: web.Request) -> tuple[dict[str, Any], web.Response | None]:
        cuerpo: dict[str, Any] = await pedido.json() if pedido.can_read_body else {}
        self.pedidos.append((pedido.method, pedido.path, cuerpo))
        self.api_keys.add(pedido.headers.get("Api-Key", ""))
        pendientes = self.fallas.get(pedido.path, [])
        if pendientes:
            return cuerpo, web.json_response({"error": {"code": "UNAVAILABLE", "message": "falla simulada"}}, status=pendientes.pop(0))
        return cuerpo, None

    # Describe un índice en el formato de la API
    def _modelo_indice(self, nombre: str) -> dict[str, Any]:
        indice = self.indices[nombre]
        return {
            "name": nombre, "dimension": indice["dimension"], "metric": "cosine", "host": f"{self.url}",
            "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}}, "status": {"ready": True, "state": "Ready"},
            "deletion_protection": "disabled", "vector_type": "dense",
        }

    # GET /indexes
    async def listar(self, pedido: web.Request) -> web.Response:
        _, error = await self._entrar(pedido)
        return error or web.json_response({"indexes": [self._modelo_indice(n) for n in self.indices]})

    # POST /indexes
    async def crear(self, pedido: web.Request) -> web.Response:
        cuerpo, error = await self._entrar(pedido)
        if error:
            return error
        self.indices[cuerpo["name"]] = {"dimension": cuerpo["dimension"]}
        return web.json_response(self._modelo_indice(cuerpo["name"]), status=201)

    # GET /indexes/{nombre}
    async def describir(self, pedido: web.Request) -> web.Response:
        _, error = await self._entrar(pedido)
        nombre = pedido.match_info["nombre"]
        if error:
            return error
        if nombre not in self.indices:
            return web.json_response({"error": {"code": "NOT_FOUND", "message": f"Index {nombre} not found"}}, status=404)
        return web.json_response(self._modelo_indice(nombre))

    # POST /embed (Pinecone Inference)
    async def embeber(self, pedido: web.Request) -> web.Response:
        cuerpo, error = await self._entrar(pedido)
        if error:
            return error
        datos = [{"vector_type": "dense", "values": self.vector(entrada["text"])} for entrada in cuerpo["inputs"]]
        return web.json_response({"model": cuerpo["model"], "vector_type": "dense", "data": datos, "usage": {"total_tokens": len(datos)}})

    # POST /vectors/upsert
    async def upsert(self, pedido: web.Request) -> web.Response:
        cuerpo, error = await self._entrar(pedido)
        if error:
            return error
        almacen = self.vectores.setdefault(("indice", cuerpo.get("namespace", "")), {})
        for vector in cuerpo["vectors"]:
            almacen[vector["id"]] = vector
        return web.json_response({"upsertedCount": len(cuerpo["vectors"])})

    # POST /query (soporta el filtro {"campo": {"$eq": valor}})
    async def consultar(self, pedido: web.Request) -> web.Response:
        cuerpo, error = await self._entrar(pedido)
        if error:
            return error
        almacen = self.vectores.get(("indice", cuerpo.get("namespace", "")), {})
        filtro: dict[str, Any] = cuerpo.get("filter") or {}
        candidatos: list[tuple[float, str]] = []
        for identificador, vector in almacen.items():
            metadata = vector.get("metadata", {})
            if any(metadata.get(campo) != condicion["$eq"] for campo, condicion in filtro.items()):
                continue
            similitud = sum(a * b for a, b in zip(cuerpo["vector"], vector["values"], strict=True))
            candidatos.append((similitud, identificador))
        candidatos.sort(key=lambda par: (-par[0], par[1]))
        coincidencias = [{"id": identificador, "score": similitud} for similitud, identificador in candidatos[: cuerpo["topK"]]]
        return web.json_response({"matches": coincidencias, "namespace": cuerpo.get("namespace", ""), "usage": {"readUnits": 1}})
