# Tests de las reglas de negocio (app/calculos.py): funciones puras con contratos Pydantic de entrada y de salida
# Cada regla se prueba en sus límites (el error típico está en el borde: 10 vs 11 días, 49.999 vs 50.000, 5 vs 10 años)

# Importa pytest para parametrizar y comprobar excepciones
import pytest
# Importa BaseModel y ValidationError: los contratos de entrada rechazan lo inválido
from pydantic import BaseModel, ValidationError

# Importa las funciones, los contratos y el registro a probar
from app.calculos import (
    CALCULOS,
    EntradaDescuento,
    EntradaDevolucion,
    EntradaEnvio,
    EntradaTotalPedido,
    EntradaVacaciones,
    calcular_descuento,
    calcular_dias_vacaciones,
    calcular_envio,
    calcular_total_pedido,
    recalcular,
    verificar_plazo_devolucion,
)


# Test 1: la escala de descuento por volumen respeta cada umbral (9, 10, 49, 50, 99 y 100 unidades)
@pytest.mark.parametrize(("cantidad", "esperado"), [(1, 0), (9, 0), (10, 5), (49, 5), (50, 10), (99, 10), (100, 15), (5000, 15)])
def test_descuento_por_volumen(cantidad: int, esperado: float) -> None:
    resultado = calcular_descuento(EntradaDescuento(cantidad=cantidad))
    assert resultado.descuento_volumen_pct == esperado
    assert resultado.descuento_mayorista_pct == 0
    assert resultado.descuento_total_pct == esperado


# Test 2: el descuento mayorista (3%) se SUMA al de volumen
def test_descuento_mayorista_se_suma() -> None:
    resultado = calcular_descuento(EntradaDescuento(cantidad=60, es_mayorista=True))
    assert (resultado.descuento_volumen_pct, resultado.descuento_mayorista_pct, resultado.descuento_total_pct) == (10, 3, 13)
    # Un mayorista que compra poco igual recibe su 3%
    assert calcular_descuento(EntradaDescuento(cantidad=2, es_mayorista=True)).descuento_total_pct == 3


# Test 3: el total del pedido aplica el descuento al precio de lista y subtotal - ahorro = total siempre cierra exacto
def test_total_pedido() -> None:
    resultado = calcular_total_pedido(EntradaTotalPedido(precio_unitario=1200, cantidad=60, descuento_total_pct=13))
    assert (resultado.subtotal, resultado.ahorro, resultado.total_productos) == (72000.0, 9360.0, 62640.0)


# Test 4: con importes que en coma flotante binaria dan error (0.1 + 0.2), el cálculo usa Decimal y cierra al centavo
@pytest.mark.parametrize(("precio", "cantidad", "pct"), [(0.1, 3, 7), (19.99, 7, 13), (1234.56, 11, 5), (0.33, 3, 33.3)])
def test_total_pedido_cierra_al_centavo(precio: float, cantidad: int, pct: float) -> None:
    resultado = calcular_total_pedido(EntradaTotalPedido(precio_unitario=precio, cantidad=cantidad, descuento_total_pct=pct))
    assert round(resultado.subtotal - resultado.ahorro, 2) == resultado.total_productos


# Test 5: el envío es gratis desde $50.000 inclusive y cuesta $2.500 por debajo
@pytest.mark.parametrize(
    ("monto", "gratis", "costo", "total"),
    [(0, False, 2500.0, 2500.0), (49999.99, False, 2500.0, 52499.99), (50000, True, 0.0, 50000.0), (62640, True, 0.0, 62640.0)],
)
def test_envio(monto: float, gratis: bool, costo: float, total: float) -> None:
    resultado = calcular_envio(EntradaEnvio(monto_compra=monto))
    assert (resultado.envio_gratis, resultado.costo_envio, resultado.total_con_envio) == (gratis, costo, total)


# Test 6: vacaciones por antigüedad, con aclaración en los límites exactos (5 y 10 años figuran en dos tramos de la política)
@pytest.mark.parametrize(
    ("anios", "dias", "con_nota"),
    [(0, 14, False), (4.9, 14, False), (5, 21, True), (7, 21, False), (10, 21, True), (10.1, 28, False), (25, 28, False)],
)
def test_dias_de_vacaciones(anios: float, dias: int, con_nota: bool) -> None:
    resultado = calcular_dias_vacaciones(EntradaVacaciones(antiguedad_anios=anios))
    assert resultado.dias_corridos == dias
    assert bool(resultado.nota) is con_nota


# Test 7: devoluciones: dentro de los 10 días se acepta; después solo con defecto de fábrica
@pytest.mark.parametrize(
    ("dias", "defecto", "dentro", "aceptada"),
    [(0, False, True, True), (10, False, True, True), (11, False, False, False), (11, True, False, True), (40, True, False, True)],
)
def test_plazo_de_devolucion(dias: int, defecto: bool, dentro: bool, aceptada: bool) -> None:
    resultado = verificar_plazo_devolucion(EntradaDevolucion(dias_desde_recepcion=dias, defecto_de_fabrica=defecto))
    assert (resultado.dentro_de_plazo, resultado.devolucion_aceptada, resultado.plazo_dias) == (dentro, aceptada, 10)
    assert resultado.motivo


# Test 8: los contratos de entrada rechazan datos imposibles (Pydantic valida lo que el LLM manda a las herramientas)
@pytest.mark.parametrize(
    ("modelo", "datos"),
    [
        (EntradaDescuento, {"cantidad": 0}),
        (EntradaDescuento, {"cantidad": "muchas"}),
        (EntradaTotalPedido, {"precio_unitario": -5, "cantidad": 1, "descuento_total_pct": 0}),
        (EntradaTotalPedido, {"precio_unitario": 100, "cantidad": 1, "descuento_total_pct": 150}),
        (EntradaEnvio, {"monto_compra": -1}),
        (EntradaVacaciones, {"antiguedad_anios": -2}),
        (EntradaDevolucion, {"dias_desde_recepcion": -1}),
    ],
)
def test_entradas_invalidas(modelo: type[BaseModel], datos: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        modelo.model_validate(datos)


# Test 9: el registro tiene los 5 cálculos y cada uno se puede recalcular desde sus entradas (lo que hace el árbitro)
def test_registro_y_recalculo() -> None:
    assert set(CALCULOS) == {"calcular_descuento", "calcular_total_pedido", "calcular_envio", "calcular_dias_vacaciones", "verificar_plazo_devolucion"}
    assert recalcular("calcular_descuento", {"cantidad": 60, "es_mayorista": True})["descuento_total_pct"] == 13
    assert recalcular("calcular_envio", {"monto_compra": 100})["costo_envio"] == 2500.0
    with pytest.raises(KeyError):
        recalcular("calcular_inexistente", {})
    with pytest.raises(ValidationError):
        recalcular("calcular_envio", {"monto_compra": "abc"})
