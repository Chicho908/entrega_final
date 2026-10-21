# Tests del verificador de importes: la respuesta final solo puede citar importes que salgan de los datos validados

# Importa lo que se prueba
from app.montos import a_numero, extraer_importes, montos_no_respaldados, numeros_en

# Datos de ejemplo: un cálculo de pedido mayorista y un fragmento de política con el umbral de envío gratis
DATOS = {
    "pregunta": "Soy mayorista y compro 60 unidades de $1.200",
    "fragmentos": [{"texto": "El envío es gratis en compras desde $50.000; si no, cuesta $2.500."}],
    "calculos": [{"herramienta": "calcular_total_pedido", "entradas": {"precio_unitario": 1200.0, "cantidad": 60}, "resultado": {"subtotal": 72000.0, "ahorro": 9360.0, "total_productos": 62640.0}}],
}


# Test 1: lee el formato argentino (puntos de miles, coma decimal)
def test_a_numero() -> None:
    assert a_numero("62.640") == 62640.0
    assert a_numero("1.234.567,50") == 1234567.5
    assert a_numero("1200") == 1200.0
    assert a_numero("2500,5") == 2500.5


# Test 2: extrae solo los importes con signo $, aunque vengan en negrita, con espacio o seguidos de puntuación
def test_extraer_importes() -> None:
    texto = "Pagás **$62.640**, ahorrás $ 9.360, el envío $0. Son 60 unidades y un 13% menos. Total: $62.640."
    assert extraer_importes(texto) == [62640.0, 9360.0, 0.0, 62640.0]


# Test 3: junta los números de los datos (de cálculos y de los textos de los fragmentos), sin contar los booleanos
def test_numeros_en() -> None:
    numeros = numeros_en(DATOS) | numeros_en({"envio_gratis": True})
    assert {62640.0, 72000.0, 9360.0, 50000.0, 2500.0, 1200.0} <= numeros
    assert 1.0 not in numeros_en({"envio_gratis": True})


# Test 4: una respuesta que cita solo importes de los datos está respaldada
def test_respuesta_respaldada() -> None:
    respuesta = "Subtotal $72.000, ahorro $9.360 (13%), total $62.640. El envío es gratis porque supera $50.000."
    assert montos_no_respaldados(respuesta, DATOS) == []


# Test 5: detecta el error real que encontramos en la prueba con el LLM: el monto de cada descuento por separado no estaba en los datos
def test_detecta_importes_inventados() -> None:
    respuesta = "Descuento por volumen: -$3.600. Descuento mayorista: -$5.760. Total: $62.640."
    assert montos_no_respaldados(respuesta, DATOS) == [3600.0, 5760.0]


# Test 6: tolera centavos de diferencia y no objeta el $0 (por ejemplo, envío gratis)
def test_tolerancia_y_cero() -> None:
    assert montos_no_respaldados("Total $62.640,00 y envío $0.", DATOS) == []
    assert montos_no_respaldados("Total $62.700.", DATOS) == [62700.0]
