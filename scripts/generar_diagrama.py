# Genera el diagrama Mermaid del grafo de agentes a partir del grafo REAL (no está dibujado a mano: si cambia la topología, el diagrama cambia solo)
# No necesita .env, API keys ni red: compilar el grafo no llama al LLM ni a Pinecone
#
# Uso:
#   python -m scripts.generar_diagrama            imprime el texto Mermaid para pegar en el README

# Importa Settings (la configuración se arma a mano: no hace falta ningún .env para dibujar el grafo)
from app.config import Settings
# Importa el armado del grafo y el grupo de LLM
from app.graph import LLMs, crear_grafo
# Importa la fábrica del LLM (construir el cliente no hace ninguna llamada de red)
from app.llm import crear_llm
# Importa el contrato de la búsqueda
from app.rag.recuperador import ResultadoBusqueda


# Recuperador de mentira: el grafo solo necesita que exista para armarse, el diagrama no ejecuta ninguna búsqueda
class RecuperadorNulo:
    # Nunca se llama al generar el diagrama
    async def buscar(self, consulta: str, k: int, categoria: str | None = None) -> ResultadoBusqueda:
        return ResultadoBusqueda(fragmentos=[], modo="hibrido")


# Compila el grafo y devuelve su diagrama en formato Mermaid
def generar_mermaid() -> str:
    configuracion = Settings(
        proveedor="anthropic", api_key="no-se-usa", modelo="no-se-usa", limite_recursion=40, max_pasos_supervisor=8,
        max_refinamientos=1, limite_iteraciones_agente=8,
    )
    llm = crear_llm(configuracion)
    grafo = crear_grafo(LLMs(supervisor=llm, investigador=llm, analista=llm, sintetizador=llm), configuracion, RecuperadorNulo()).compile()
    return grafo.get_graph().draw_mermaid()


if __name__ == "__main__":
    print(generar_mermaid())
