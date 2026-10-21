# Módulo de montos: verifica que los importes en pesos de la respuesta final salgan de los datos validados
# El LLM redacta, pero no puede inventar cifras: si escribe un importe que no está en los cálculos ni en los fragmentos de política, se rechaza y se reintenta

# Importa re para buscar importes en el texto
import re
# Importa Any para tipar los datos recorridos
from typing import Any

# Un importe en pesos con signo $ (acepta "$62.640", "$ 62.640,50", "$1200"): captura solo el número
PATRON_IMPORTE = re.compile(r"\$\s*(\d+(?:\.\d{3})*(?:,\d+)?|\d+(?:\.\d+)?)")
# Cualquier número escrito con formato argentino (puntos de miles y coma decimal), para leer los fragmentos de política
PATRON_NUMERO = re.compile(r"\d+(?:\.\d{3})*(?:,\d+)?")
# Diferencia máxima para considerar iguales dos importes (centavos)
TOLERANCIA = 0.01


# Convierte un número escrito en formato argentino ("62.640", "62.640,50", "1200") a float
def a_numero(texto: str) -> float:
    # Los puntos son separadores de miles y la coma es el decimal
    if "," in texto or re.fullmatch(r"\d{1,3}(\.\d{3})+", texto):
        return float(texto.replace(".", "").replace(",", "."))
    return float(texto)


# Importes en pesos (con signo $) que aparecen en un texto
def extraer_importes(texto: str) -> list[float]:
    return [a_numero(coincidencia) for coincidencia in PATRON_IMPORTE.findall(texto)]


# Junta todos los números de una estructura de datos: los numéricos tal cual y los que aparecen dentro de textos (por ejemplo, los fragmentos de política)
def numeros_en(valor: Any) -> set[float]:
    numeros: set[float] = set()
    # bool es subclase de int: se excluye para que True no cuente como el número 1
    if isinstance(valor, (int, float)) and not isinstance(valor, bool):
        numeros.add(float(valor))
    elif isinstance(valor, str):
        numeros.update(a_numero(coincidencia) for coincidencia in PATRON_NUMERO.findall(valor))
    elif isinstance(valor, dict):
        for elemento in valor.values():
            numeros |= numeros_en(elemento)
    elif isinstance(valor, (list, tuple)):
        for elemento in valor:
            numeros |= numeros_en(elemento)
    return numeros


# Importes de la respuesta que NO figuran en los datos validados (lista vacía = todo está respaldado)
def montos_no_respaldados(respuesta: str, datos: dict[str, Any]) -> list[float]:
    permitidos = numeros_en(datos)
    return [
        importe
        for importe in extraer_importes(respuesta)
        if importe != 0 and not any(abs(importe - permitido) <= TOLERANCIA for permitido in permitidos)
    ]
