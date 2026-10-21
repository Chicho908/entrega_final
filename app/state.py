# Módulo de estado: define el esquema de datos compartido entre el supervisor y los especialistas
# Regla de oro: cada campo del estado tiene UN solo dueño (un nodo que lo escribe); los demás solo lo leen

# Importa add para usarlo como reducer: concatena las listas en vez de reemplazarlas
from operator import add
# Importa los tipos para anotar el estado y los contratos
from typing import Annotated, Any, Literal

# Importa HumanMessage para armar el mensaje inicial del usuario
from langchain_core.messages import HumanMessage
# Importa MessagesState: ya trae el campo "messages" con el reducer add_messages
from langgraph.graph import MessagesState
# Importa BaseModel y Field para definir contratos validados (los especialistas no pueden devolver cualquier cosa)
from pydantic import BaseModel, Field

# Importa los nombres de los cálculos disponibles
from app.calculos import NombreCalculo

# Nombres de los nodos a los que puede ir el supervisor (se usa también para tipar las aristas condicionales)
Destino = Literal["investigador", "analista", "sintetizador"]


# Un fragmento de política recuperado por el investigador: viene siempre del corpus real (PDF de EmpresaFlomat) y lleva su procedencia
class FragmentoPolitica(BaseModel):
    # Identificador estable del fragmento (hash de fuente + página + texto): permite comprobar que nadie lo alteró
    fragment_id: str = Field(min_length=8, max_length=64)
    # PDF de origen (procedencia: de dónde salió el dato)
    fuente: str = Field(pattern=r"^politica_[a-z]+\.pdf$")
    # Página del PDF (empieza en 1)
    pagina: int = Field(ge=1)
    # Categoría de la política (descuentos, devoluciones, envios o vacaciones)
    categoria: str = Field(min_length=1)
    # Título de la sección dentro del documento
    seccion: str = ""
    # Texto del fragmento
    texto: str = Field(min_length=1)
    # Puntaje de relevancia de la fusión RRF (más alto = más relevante)
    score: float = Field(ge=0)


# Contrato de salida del investigador: debe traer al menos un fragmento
class ResultadoInvestigacion(BaseModel):
    # Instrucción que recibió el investigador
    consulta: str = Field(min_length=3)
    # Fragmentos recuperados (mínimo uno: sin datos no hay nada en lo que basar una respuesta)
    fragmentos: list[FragmentoPolitica] = Field(min_length=1)
    # Resumen breve escrito por el LLM del investigador
    resumen: str = Field(min_length=1)


# Un cálculo que hizo una herramienta del analista, con sus entradas y su resultado exacto
class CalculoRealizado(BaseModel):
    # Herramienta que lo calculó (solo las del registro de calculos.py)
    herramienta: NombreCalculo
    # Argumentos con los que se llamó a la herramienta
    entradas: dict[str, Any]
    # Resultado que devolvió la herramienta
    resultado: dict[str, Any]
    # Política (PDF) en la que se basa el cálculo
    politica: str = Field(pattern=r"^politica_[a-z]+\.pdf$")


# Contrato de salida del analista: números calculados por herramientas + conclusiones del LLM
class ResultadoAnalisis(BaseModel):
    # Todos los cálculos que hizo (mínimo uno: si no hizo ninguno, no analizó nada)
    calculos: list[CalculoRealizado] = Field(min_length=1)
    # Conclusiones escritas por el LLM del analista
    conclusiones: str = Field(min_length=1)


# Decisión estructurada del supervisor: el LLM está obligado a elegir uno de los nodos válidos
class Decision(BaseModel):
    # A qué nodo ir ahora (el Literal impide inventar nombres de nodos)
    siguiente: Destino = Field(description="Próximo nodo: investigador, analista o sintetizador.")
    # Instrucción AUTOCONTENIDA para el especialista (campo obligatorio: sin valor por defecto, para que los proveedores con esquema estricto lo acepten)
    instruccion: str = Field(description="Instrucción autocontenida para el especialista; vacía si el destino es sintetizador.")
    # Razón breve de la decisión (queda en la traza para auditar)
    razon: str = Field(min_length=1, description="Una frase que explica por qué se eligió ese paso.")


# Estado compartido del orquestador
class EstadoOrquestador(MessagesState):
    # --- Entrada (la completa la API al crear la tarea) ---
    # Destinatario al que hay que enviar el informe ("" = solo responder, sin efectos secundarios)
    notificar_a: str
    # --- Escribe el supervisor ---
    # Próximo nodo elegido (lo lee la arista condicional)
    siguiente: str
    # Instrucción específica para el especialista que va a trabajar
    instruccion: str
    # Cantidad de decisiones que ya tomó el supervisor (contador anti "supervisor infinito")
    pasos_supervisor: int
    # Historial auditable de decisiones (propuesta del LLM, decisión final y guardarraíl aplicado)
    decisiones: Annotated[list[dict[str, Any]], add]
    # --- Escribe el investigador ---
    # Resultado crudo de la investigación (se valida en el nodo de validación)
    investigacion: dict[str, Any]
    # Veces que se ejecutó el investigador
    intentos_investigador: int
    # --- Escribe el analista ---
    # Resultado crudo del análisis (se valida en el nodo de validación)
    analisis: dict[str, Any]
    # Veces que se ejecutó el analista
    intentos_analista: int
    # --- Escriben los dos especialistas (de a uno por vez) ---
    # Último especialista que trabajó (le dice a la validación qué revisar)
    ultimo_agente: str
    # Registro de quién contribuyó con qué (el reducer add concatena, nunca pisa)
    contribuciones: Annotated[list[dict[str, Any]], add]
    # --- Escribe la validación ---
    # True si la investigación pasó el contrato y las reglas de negocio
    investigacion_valida: bool
    # True si el análisis pasó el contrato y las reglas de negocio
    analisis_valido: bool
    # Motivo específico del rechazo de la investigación ("" si está bien)
    error_investigacion: str
    # Motivo específico del rechazo del análisis ("" si está bien)
    error_analisis: str
    # --- Escribe el sintetizador ---
    # Respuesta final para el usuario
    respuesta_final: str
    # PDF que sustentan la respuesta (trazabilidad de la procedencia)
    fuentes: list[str]
    # --- Escribe la aprobación humana (solo en tareas críticas) ---
    # Decisión del revisor: {"estado": "aprobada" | "rechazada", "comentario": str}
    aprobacion: dict[str, Any]
    # --- Escribe el despacho (la única acción con efectos secundarios) ---
    # Resultado del envío del informe ("" si la tarea no era crítica)
    despacho: str


# Arma el estado inicial de una ejecución: todos los campos con su valor neutro
def estado_inicial(pregunta: str, notificar_a: str = "") -> EstadoOrquestador:
    return {
        "messages": [HumanMessage(content=pregunta)],
        "notificar_a": notificar_a,
        "siguiente": "",
        "instruccion": "",
        "pasos_supervisor": 0,
        "decisiones": [],
        "investigacion": {},
        "intentos_investigador": 0,
        "analisis": {},
        "intentos_analista": 0,
        "ultimo_agente": "",
        "contribuciones": [],
        "investigacion_valida": False,
        "analisis_valido": False,
        "error_investigacion": "",
        "error_analisis": "",
        "respuesta_final": "",
        "fuentes": [],
        "aprobacion": {},
        "despacho": "",
    }
