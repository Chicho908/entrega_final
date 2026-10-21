# Módulo de validación: el "árbitro" determinista. Revisa lo que entregó cada especialista ANTES de que el supervisor decida
# No usa LLM: son reglas de código, así que el veredicto es siempre el mismo y el mensaje de error es específico

# Importa Any para tipar los diccionarios de estado
from typing import Any

# Importa ValidationError para convertir los errores de Pydantic en texto claro
from pydantic import ValidationError

# Importa el recálculo de los cálculos para comprobar los resultados del analista
from app.calculos import recalcular
# Importa el corpus: las fuentes válidas y el cálculo del id de un fragmento
from app.rag.corpus import CATEGORIAS, id_fragmento
# Importa los contratos
from app.state import EstadoOrquestador, ResultadoAnalisis, ResultadoInvestigacion


# Convierte un ValidationError de Pydantic en una frase corta y accionable
def resumir_error(error: ValidationError) -> str:
    # Una parte por cada campo inválido: "campo: mensaje"
    partes = [f"{'.'.join(str(p) for p in e['loc']) or 'resultado'}: {e['msg']}" for e in error.errors()]
    # Une las partes (máximo 4 para no llenar el prompt del reintento)
    return "; ".join(partes[:4])


# Valida la investigación: contrato + procedencia de cada fragmento. Devuelve "" si está bien o el motivo del rechazo
def validar_investigacion(investigacion: dict[str, Any]) -> str:
    # Si el especialista no entregó nada (falló o no encontró datos)
    if not investigacion:
        return "el investigador no entregó ningún resultado"
    # Valida el contrato con Pydantic
    try:
        resultado = ResultadoInvestigacion.model_validate(investigacion)
    except ValidationError as error:
        # Sin fragmentos hay dos causas muy distintas: la fuente se cayó (falla técnica, conviene reintentar) o no hay datos (no conviene insistir)
        if not investigacion.get("fragmentos") and investigacion.get("fuente_caida"):
            return "la búsqueda en las políticas no respondió (falla técnica, no falta de datos): reintentá la misma búsqueda; si vuelve a fallar, no hay forma de responder y hay que informar la falla"
        if not investigacion.get("fragmentos"):
            return "la búsqueda no recuperó ningún fragmento: probá con otras palabras del mismo tema; si las políticas no cubren el tema, no hay datos y no hay que inventarlos"
        return resumir_error(error)
    # Reglas de procedencia: cada fragmento tiene que venir de un documento real del corpus y no estar alterado
    for fragmento in resultado.fragmentos:
        if fragmento.fuente not in CATEGORIAS.values() or CATEGORIAS.get(fragmento.categoria) != fragmento.fuente:
            return f"el fragmento '{fragmento.fragment_id}' cita una fuente o categoría que no existe en el corpus ({fragmento.fuente}, {fragmento.categoria})"
        if id_fragmento(fragmento.fuente, fragmento.pagina, fragmento.texto) != fragmento.fragment_id:
            return f"el fragmento '{fragmento.fragment_id}' fue alterado: su contenido no coincide con su identificador"
    return ""


# Comprueba que los cálculos encadenados sean coherentes entre sí. Devuelve "" si lo son o el motivo del rechazo
def _validar_encadenado(calculos: list[dict[str, Any]]) -> str:
    # Último cálculo de cada herramienta (si el analista repitió uno para corregirlo, vale el más reciente)
    ultimos = {c["herramienta"]: c for c in calculos}
    descuento = ultimos.get("calcular_descuento")
    pedido = ultimos.get("calcular_total_pedido")
    envio = ultimos.get("calcular_envio")
    # El total del pedido debe usar el descuento que calculó calcular_descuento
    if descuento and pedido and pedido["entradas"].get("descuento_total_pct") != descuento["resultado"]["descuento_total_pct"]:
        return (
            f"calcular_total_pedido usó un descuento de {pedido['entradas'].get('descuento_total_pct')}%, pero calcular_descuento dio "
            f"{descuento['resultado']['descuento_total_pct']}%: usá el descuento_total_pct que devolvió calcular_descuento"
        )
    # El envío se decide sobre el monto ya con descuento
    if pedido and envio and envio["entradas"].get("monto_compra") != pedido["resultado"]["total_productos"]:
        return (
            f"calcular_envio se calculó sobre {envio['entradas'].get('monto_compra')}, pero el total de los productos con descuento es "
            f"{pedido['resultado']['total_productos']}: el envío se decide sobre el monto ya con descuento"
        )
    return ""


# Valida el análisis: contrato + cada cálculo recalculado + respaldo en la investigación + coherencia entre cálculos
def validar_analisis(analisis: dict[str, Any], investigacion: dict[str, Any]) -> str:
    # Si el especialista no entregó nada
    if not analisis:
        return "el analista no entregó ningún resultado"
    # Valida el contrato con Pydantic (incluye: al menos un cálculo)
    try:
        resultado = ResultadoAnalisis.model_validate(analisis)
    except ValidationError as error:
        if not analisis.get("calculos"):
            return "el analista no ejecutó ningún cálculo: usá tus herramientas con los datos de la orden, o explicá qué dato falta"
        return f"faltan o son inválidos estos datos del análisis → {resumir_error(error)}"
    # Datos de la investigación contra los que se contrasta (ya validados antes de llegar acá)
    try:
        fragmentos = ResultadoInvestigacion.model_validate(investigacion).fragmentos
    # Sin una investigación válida no hay contra qué contrastar (el supervisor no debería llegar acá)
    except ValidationError:
        return "no se puede validar el análisis porque no hay una investigación válida"
    fuentes_recuperadas = {f.fuente for f in fragmentos}
    for calculo in resultado.calculos:
        # Regla de recálculo: el árbitro vuelve a ejecutar el cálculo con las mismas entradas y el resultado tiene que ser idéntico
        try:
            esperado = recalcular(calculo.herramienta, calculo.entradas)
        except (ValidationError, KeyError, TypeError):
            return f"el cálculo {calculo.herramienta} tiene entradas inválidas: {calculo.entradas}"
        if esperado != calculo.resultado:
            return f"el resultado de {calculo.herramienta} no coincide con el recalculado ({calculo.resultado} contra {esperado})"
        # Regla de respaldo: no se calcula con una política que el investigador no recuperó
        if calculo.politica not in fuentes_recuperadas:
            return f"el cálculo {calculo.herramienta} se basa en {calculo.politica}, pero el investigador no recuperó esa política: hay que investigarla primero"
    # Regla de coherencia entre cálculos encadenados
    return _validar_encadenado([c.model_dump() for c in resultado.calculos])


# Nodo de validación: revisa al último especialista que trabajó y deja el veredicto en el estado
async def nodo_validacion(state: EstadoOrquestador) -> dict[str, Any]:
    # Averigua a quién hay que revisar
    agente = state["ultimo_agente"]
    # Revisa la investigación
    if agente == "investigador":
        error = validar_investigacion(state.get("investigacion", {}))
        cambios: dict[str, Any] = {"investigacion_valida": error == "", "error_investigacion": error}
    # Revisa el análisis (contrastándolo con la investigación)
    else:
        error = validar_analisis(state.get("analisis", {}), state.get("investigacion", {}))
        cambios = {"analisis_valido": error == "", "error_analisis": error}
    # Deja el veredicto también en el registro de contribuciones
    cambios["contribuciones"] = [
        {"agente": "validacion", "sobre": agente, "valido": error == "", "resumen": error or "cumple el contrato y las reglas de negocio"}
    ]
    return cambios
