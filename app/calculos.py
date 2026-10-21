# Módulo de cálculos: las reglas numéricas de las políticas de EmpresaFlomat como funciones puras (sin LLM, sin red, sin estado)
# Cada cálculo tiene un contrato Pydantic de entrada y otro de salida: es lo que validan las herramientas del analista
# y también lo que usa el árbitro (validation.py) para RECALCULAR cada resultado y comprobar que nadie lo alteró

# Importa Callable para tipar la función de cada cálculo
from collections.abc import Callable
# Importa dataclass para definir cada cálculo del registro como un objeto inmutable
from dataclasses import dataclass
# Importa Decimal y ROUND_HALF_UP para calcular dinero sin errores de coma flotante
from decimal import ROUND_HALF_UP, Decimal
# Importa Any y Literal para tipar el registro de cálculos
from typing import Any, Final, Literal

# Importa BaseModel y Field para definir los contratos de entrada y de salida
from pydantic import BaseModel, Field

# Nombre del PDF de cada política (es la "fuente" que cita el RAG)
POLITICA_DESCUENTOS: Final = "politica_descuentos.pdf"
POLITICA_ENVIOS: Final = "politica_envios.pdf"
POLITICA_DEVOLUCIONES: Final = "politica_devoluciones.pdf"
POLITICA_VACACIONES: Final = "politica_vacaciones.pdf"

# Escala de descuento por volumen: (desde cuántas unidades, porcentaje), de mayor a menor
ESCALA_VOLUMEN: tuple[tuple[int, int], ...] = ((100, 15), (50, 10), (10, 5))
# Descuento adicional fijo para clientes mayoristas (se suma al de volumen)
DESCUENTO_MAYORISTA_PCT = 3
# Monto de compra a partir del cual el envío es gratis, y costo fijo del envío por debajo de ese monto
ENVIO_GRATIS_DESDE = Decimal("50000")
COSTO_ENVIO = Decimal("2500")
# Días corridos que tiene el cliente para devolver un producto sin justificar el motivo
PLAZO_DEVOLUCION_DIAS = 10
# Vacaciones por antigüedad: (hasta cuántos años, exclusive; días corridos). El último tramo es "más de 10 años"
DIAS_VACACIONES_HASTA_5 = 14
DIAS_VACACIONES_5_A_10 = 21
DIAS_VACACIONES_MAS_DE_10 = 28


# Redondea un importe a 2 decimales (mitad hacia arriba, como en facturación) y lo devuelve como float
def _redondear(valor: Decimal) -> float:
    return float(valor.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


# ---------- 1. Descuento por volumen y mayorista ----------


# Entrada de calcular_descuento
class EntradaDescuento(BaseModel):
    # Unidades del MISMO producto en el pedido
    cantidad: int = Field(ge=1, le=1_000_000, description="Unidades del mismo producto en el pedido.")
    # True si el cliente está registrado como mayorista
    es_mayorista: bool = Field(default=False, description="True si el cliente es mayorista registrado.")


# Salida de calcular_descuento
class SalidaDescuento(BaseModel):
    # Política en la que se basa el cálculo
    politica: Literal["politica_descuentos.pdf"] = POLITICA_DESCUENTOS
    # Porcentaje por cantidad (0, 5, 10 o 15)
    descuento_volumen_pct: float
    # Porcentaje mayorista (0 o 3)
    descuento_mayorista_pct: float
    # Suma de ambos: es el porcentaje que se aplica al precio de lista
    descuento_total_pct: float


# Calcula el descuento que corresponde según la cantidad y el tipo de cliente
def calcular_descuento(entrada: EntradaDescuento) -> SalidaDescuento:
    # Busca el primer umbral de la escala que la cantidad alcanza (la escala está ordenada de mayor a menor)
    volumen = next((pct for desde, pct in ESCALA_VOLUMEN if entrada.cantidad >= desde), 0)
    mayorista = DESCUENTO_MAYORISTA_PCT if entrada.es_mayorista else 0
    return SalidaDescuento(
        descuento_volumen_pct=float(volumen),
        descuento_mayorista_pct=float(mayorista),
        descuento_total_pct=float(volumen + mayorista),
    )


# ---------- 2. Total del pedido ----------


# Entrada de calcular_total_pedido
class EntradaTotalPedido(BaseModel):
    # Precio de lista de UNA unidad
    precio_unitario: float = Field(gt=0, le=1_000_000_000, description="Precio de lista de una unidad, en pesos.")
    # Unidades compradas
    cantidad: int = Field(ge=1, le=1_000_000, description="Unidades compradas.")
    # Porcentaje de descuento total (el que devolvió calcular_descuento)
    descuento_total_pct: float = Field(ge=0, le=100, description="Descuento total en porcentaje (0 a 100).")


# Salida de calcular_total_pedido
class SalidaTotalPedido(BaseModel):
    # Política en la que se basa el cálculo
    politica: Literal["politica_descuentos.pdf"] = POLITICA_DESCUENTOS
    # Precio de lista por cantidad, sin descuento
    subtotal: float
    # Cuánto se ahorra el cliente
    ahorro: float
    # Total de los productos ya con descuento (sin envío)
    total_productos: float


# Calcula el total de los productos con el descuento aplicado sobre el precio de lista
def calcular_total_pedido(entrada: EntradaTotalPedido) -> SalidaTotalPedido:
    # Usa Decimal(str(...)) para no arrastrar el error binario de los float
    subtotal = Decimal(str(entrada.precio_unitario)) * entrada.cantidad
    ahorro = subtotal * Decimal(str(entrada.descuento_total_pct)) / Decimal(100)
    # Se redondea cada importe a centavos y el total se obtiene de los importes ya redondeados: así subtotal - ahorro = total siempre cierra exacto
    subtotal_redondeado = Decimal(str(_redondear(subtotal)))
    ahorro_redondeado = Decimal(str(_redondear(ahorro)))
    return SalidaTotalPedido(
        subtotal=float(subtotal_redondeado),
        ahorro=float(ahorro_redondeado),
        total_productos=_redondear(subtotal_redondeado - ahorro_redondeado),
    )


# ---------- 3. Envío ----------


# Entrada de calcular_envio
class EntradaEnvio(BaseModel):
    # Monto de la compra (los productos, ya con descuentos, sin envío)
    monto_compra: float = Field(ge=0, le=1_000_000_000, description="Monto de la compra en pesos, con descuentos y sin envío.")


# Salida de calcular_envio
class SalidaEnvio(BaseModel):
    # Política en la que se basa el cálculo
    politica: Literal["politica_envios.pdf"] = POLITICA_ENVIOS
    # True si el envío es gratis
    envio_gratis: bool
    # Costo del envío (0 si es gratis)
    costo_envio: float
    # Monto de la compra más el envío
    total_con_envio: float


# Calcula el costo de envío según el monto de la compra
def calcular_envio(entrada: EntradaEnvio) -> SalidaEnvio:
    monto = Decimal(str(entrada.monto_compra))
    gratis = monto >= ENVIO_GRATIS_DESDE
    costo = Decimal(0) if gratis else COSTO_ENVIO
    return SalidaEnvio(envio_gratis=gratis, costo_envio=_redondear(costo), total_con_envio=_redondear(monto + costo))


# ---------- 4. Vacaciones ----------


# Entrada de calcular_dias_vacaciones
class EntradaVacaciones(BaseModel):
    # Años de antigüedad del empleado (puede tener decimales: 7.5)
    antiguedad_anios: float = Field(ge=0, le=60, description="Años de antigüedad del empleado.")


# Salida de calcular_dias_vacaciones
class SalidaVacaciones(BaseModel):
    # Política en la que se basa el cálculo
    politica: Literal["politica_vacaciones.pdf"] = POLITICA_VACACIONES
    # Días corridos de vacaciones anuales
    dias_corridos: int
    # Tramo de antigüedad aplicado
    tramo: str
    # Aclaración cuando la antigüedad cae justo en el límite entre dos tramos (la política nombra el límite en los dos)
    nota: str = ""


# Calcula los días de vacaciones anuales según la antigüedad
def calcular_dias_vacaciones(entrada: EntradaVacaciones) -> SalidaVacaciones:
    anios = entrada.antiguedad_anios
    # Menos de 5 años
    if anios < 5:
        return SalidaVacaciones(dias_corridos=DIAS_VACACIONES_HASTA_5, tramo="0 a 5 años")
    # Hasta 10 años inclusive (más de 10 es el último tramo)
    if anios <= 10:
        # Los límites exactos (5 y 10) figuran en dos tramos de la política: se aplica el más favorable al empleado
        nota = "La antigüedad cae justo en el límite entre dos tramos de la política; se aplicó el tramo más favorable al empleado." if anios in (5, 10) else ""
        return SalidaVacaciones(dias_corridos=DIAS_VACACIONES_5_A_10, tramo="5 a 10 años", nota=nota)
    return SalidaVacaciones(dias_corridos=DIAS_VACACIONES_MAS_DE_10, tramo="más de 10 años")


# ---------- 5. Devoluciones ----------


# Entrada de verificar_plazo_devolucion
class EntradaDevolucion(BaseModel):
    # Días corridos que pasaron desde que el cliente recibió el producto
    dias_desde_recepcion: int = Field(ge=0, le=3650, description="Días corridos desde que el cliente recibió el producto.")
    # True si el producto llegó con un defecto de fábrica
    defecto_de_fabrica: bool = Field(default=False, description="True si el producto llegó con defecto de fábrica.")


# Salida de verificar_plazo_devolucion
class SalidaDevolucion(BaseModel):
    # Política en la que se basa el cálculo
    politica: Literal["politica_devoluciones.pdf"] = POLITICA_DEVOLUCIONES
    # True si todavía está dentro de los 10 días corridos
    dentro_de_plazo: bool
    # True si la devolución se acepta (dentro de plazo, o fuera de plazo pero con defecto de fábrica)
    devolucion_aceptada: bool
    # Plazo vigente en días corridos
    plazo_dias: int
    # Explicación corta del resultado
    motivo: str


# Verifica si una devolución está dentro del plazo (no evalúa las condiciones físicas del producto: eso lo decide el depósito)
def verificar_plazo_devolucion(entrada: EntradaDevolucion) -> SalidaDevolucion:
    dentro = entrada.dias_desde_recepcion <= PLAZO_DEVOLUCION_DIAS
    if dentro:
        motivo = f"Está dentro del plazo de {PLAZO_DEVOLUCION_DIAS} días corridos."
    elif entrada.defecto_de_fabrica:
        motivo = f"Pasó el plazo de {PLAZO_DEVOLUCION_DIAS} días, pero se acepta por defecto de fábrica."
    else:
        motivo = f"Pasó el plazo de {PLAZO_DEVOLUCION_DIAS} días corridos y no hay defecto de fábrica."
    return SalidaDevolucion(
        dentro_de_plazo=dentro,
        devolucion_aceptada=dentro or entrada.defecto_de_fabrica,
        plazo_dias=PLAZO_DEVOLUCION_DIAS,
        motivo=motivo,
    )


# ---------- Registro de cálculos ----------

# Nombres de los cálculos disponibles (son también los nombres de las herramientas del analista)
NombreCalculo = Literal[
    "calcular_descuento",
    "calcular_total_pedido",
    "calcular_envio",
    "calcular_dias_vacaciones",
    "verificar_plazo_devolucion",
]

# Un cálculo del registro: su contrato de entrada, su contrato de salida y la función que lo resuelve
@dataclass(frozen=True)
class Calculo:
    # Modelo Pydantic de los argumentos
    entrada: type[BaseModel]
    # Modelo Pydantic del resultado
    salida: type[BaseModel]
    # Función pura que calcula (recibe el modelo de entrada y devuelve el de salida)
    funcion: Callable[[Any], BaseModel]


# Registro de cálculos: el nombre de cada uno es también el nombre de su herramienta en el analista
CALCULOS: dict[str, Calculo] = {
    "calcular_descuento": Calculo(EntradaDescuento, SalidaDescuento, calcular_descuento),
    "calcular_total_pedido": Calculo(EntradaTotalPedido, SalidaTotalPedido, calcular_total_pedido),
    "calcular_envio": Calculo(EntradaEnvio, SalidaEnvio, calcular_envio),
    "calcular_dias_vacaciones": Calculo(EntradaVacaciones, SalidaVacaciones, calcular_dias_vacaciones),
    "verificar_plazo_devolucion": Calculo(EntradaDevolucion, SalidaDevolucion, verificar_plazo_devolucion),
}


# Recalcula un cálculo a partir de sus entradas y devuelve la salida como diccionario (lo usa el árbitro para comprobar resultados)
# Lanza ValidationError (entradas inválidas) o KeyError (cálculo desconocido)
def recalcular(herramienta: str, entradas: dict[str, Any]) -> dict[str, Any]:
    calculo = CALCULOS[herramienta]
    return calculo.funcion(calculo.entrada.model_validate(entradas)).model_dump()
