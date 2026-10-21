# Pruebas del sistema: 5 escenarios reales contra la API ya levantada (con LLM, Pinecone y Redis de verdad)
# Cada escenario deja su traza en Phoenix; el job_id que se imprime es el session.id de esa traza (así se encuentra para la captura de pantalla)
#
# Uso (con el sistema corriendo, ver README):
#   python -m scripts.pruebas_sistema
#   python -m scripts.pruebas_sistema --url http://localhost:8000 --timeout 180

# Importa argparse para leer los parámetros de la línea de comandos
import argparse
# Importa asyncio para correr el cliente asíncrono y esperar entre consultas
import asyncio
# Importa sys para devolver un código de salida distinto de 0 si algo falla
import sys
# Importa time para medir cuánto tarda cada escenario
import time
# Importa Callable y dataclass para describir cada escenario
from collections.abc import Callable
from dataclasses import dataclass
# Importa Any para tipar el cuerpo JSON de las respuestas
from typing import Any

# Importa httpx: cliente HTTP asíncrono
import httpx

# Estados finales de un trabajo
FINALES = {"DONE", "FAILED"}


# Un escenario de prueba: qué se pregunta y qué se espera de la respuesta
@dataclass(frozen=True)
class Escenario:
    # Nombre corto (aparece en el resumen)
    nombre: str
    # Qué demuestra este escenario
    objetivo: str
    # Pregunta que se envía a POST /tasks
    pregunta: str
    # Destinatario del informe: si tiene valor, la tarea es crítica y se pausa esperando aprobación humana
    notificar_a: str
    # Revisa el trabajo terminado y devuelve la lista de problemas encontrados (vacía = pasó)
    verificar: Callable[[dict[str, Any]], list[str]]


# Resultado de un escenario
@dataclass
class Resultado:
    # Escenario ejecutado
    escenario: Escenario
    # Identificador del trabajo (= session.id de la traza en Phoenix)
    job_id: str
    # Segundos que tardó
    segundos: float
    # Problemas encontrados (vacío = pasó)
    problemas: list[str]


# Deja el texto en minúsculas y sin separadores de miles ni de decimales, para buscar cifras sin depender del formato ("$62.640" → "$62640")
def sin_separadores(texto: str) -> str:
    return texto.lower().replace(".", "").replace(",", "").replace(" ", "")


# Escenario 1: consulta pura → solo RAG (el supervisor no llama al analista porque no hay nada que calcular)
def verificar_consulta(trabajo: dict[str, Any]) -> list[str]:
    problemas = verificar_terminado(trabajo)
    if "10" not in trabajo.get("resultado", ""):
        problemas.append("la respuesta no menciona el plazo de 10 días")
    if "politica_devoluciones.pdf" not in trabajo.get("fuentes", []):
        problemas.append("no cita politica_devoluciones.pdf como fuente")
    return problemas


# Escenario 2: pedido mayorista → RAG + 3 cálculos encadenados (descuento → total → envío)
def verificar_pedido(trabajo: dict[str, Any]) -> list[str]:
    problemas = verificar_terminado(trabajo)
    texto = trabajo.get("resultado", "")
    if "62640" not in sin_separadores(texto):
        problemas.append("el total correcto es $62.640 (60 u. x $1.200, 13% de descuento) y la respuesta no lo menciona")
    if "gratis" not in texto.lower():
        problemas.append("el envío es gratis (el total con descuento supera $50.000) y la respuesta no lo dice")
    for fuente in ("politica_descuentos.pdf", "politica_envios.pdf"):
        if fuente not in trabajo.get("fuentes", []):
            problemas.append(f"no cita {fuente} como fuente")
    return problemas


# Escenario 3: pregunta con dos partes (un dato de la política y un cálculo por antigüedad)
def verificar_vacaciones(trabajo: dict[str, Any]) -> list[str]:
    problemas = verificar_terminado(trabajo)
    texto = trabajo.get("resultado", "")
    if "21" not in texto:
        problemas.append("a 7 años de antigüedad corresponden 21 días y la respuesta no los menciona")
    if "15" not in texto:
        problemas.append("la política pide 15 días de anticipación y la respuesta no lo menciona")
    if "politica_vacaciones.pdf" not in trabajo.get("fuentes", []):
        problemas.append("no cita politica_vacaciones.pdf como fuente")
    return problemas


# Escenario 5: pregunta fuera del dominio → el sistema tiene que decir que no sabe, sin inventar y sin citar fuentes
def verificar_fuera_de_dominio(trabajo: dict[str, Any]) -> list[str]:
    problemas = verificar_terminado(trabajo)
    if trabajo.get("fuentes"):
        problemas.append(f"citó fuentes ({trabajo['fuentes']}) para una pregunta que las políticas no cubren: eso es una respuesta inventada")
    return problemas


# Todo trabajo bueno termina en DONE, con respuesta y sin error
def verificar_terminado(trabajo: dict[str, Any]) -> list[str]:
    if trabajo.get("status") != "DONE":
        return [f"terminó en {trabajo.get('status')}: {trabajo.get('error')}"]
    if not trabajo.get("resultado"):
        return ["terminó sin respuesta"]
    return []


# Escenario 4 (tarea crítica): se verifica en dos momentos, la pausa y el cierre
def verificar_envio(trabajo: dict[str, Any]) -> list[str]:
    problemas = verificar_pedido(trabajo)
    if "enviado" not in (trabajo.get("despacho") or ""):
        problemas.append(f"el informe debería haberse enviado tras la aprobación y el despacho dice: {trabajo.get('despacho')!r}")
    return problemas


# Los 5 escenarios (cubren: RAG solo, RAG + cálculos, pregunta compuesta, aprobación humana y honestidad ante lo desconocido)
ESCENARIOS: list[Escenario] = [
    Escenario(
        "1. Consulta de política (solo RAG)",
        "El supervisor investiga y cierra sin llamar al analista",
        "¿Cuántos días corridos tiene un cliente para devolver un producto?",
        "",
        verificar_consulta,
    ),
    Escenario(
        "2. Pedido mayorista (RAG + cálculos encadenados)",
        "Investigador → analista (descuento, total, envío) → validación por recálculo → síntesis",
        "Soy cliente mayorista y compro 60 unidades de $1.200 cada una. ¿Cuánto pago en total con descuentos y envío?",
        "",
        verificar_pedido,
    ),
    Escenario(
        "3. Pregunta compuesta (política + cálculo)",
        "Una sola pregunta que necesita dos fuentes del mismo documento y un cálculo",
        "Tengo 7 años de antigüedad. ¿Cuántos días de vacaciones me corresponden y con cuánta anticipación tengo que pedirlos?",
        "",
        verificar_vacaciones,
    ),
    Escenario(
        "4. Tarea crítica con aprobación humana",
        "El grafo se pausa (interrupt), un humano aprueba y recién ahí se envía el informe",
        "Soy cliente mayorista y compro 60 unidades de $1.200 cada una. ¿Cuánto pago en total con descuentos y envío?",
        "gerencia@empresa.com",
        verificar_envio,
    ),
    Escenario(
        "5. Pregunta fuera de las políticas",
        "El sistema reconoce que no tiene la información y no inventa ni cita fuentes",
        "¿Cuántos días de teletrabajo por semana tiene un empleado?",
        "",
        verificar_fuera_de_dominio,
    ),
]


# Consulta el trabajo cada "pausa_s" segundos hasta que llegue a uno de los estados esperados o se agote el tiempo
async def esperar(http: httpx.AsyncClient, job_id: str, estados: set[str], timeout_s: float, pausa_s: float) -> dict[str, Any]:
    limite = time.monotonic() + timeout_s
    while True:
        respuesta = await http.get(f"/tasks/{job_id}")
        respuesta.raise_for_status()
        trabajo: dict[str, Any] = respuesta.json()
        if trabajo["status"] in estados or time.monotonic() > limite:
            return trabajo
        await asyncio.sleep(pausa_s)


# Corre un escenario completo por la API: crea la tarea, espera, aprueba si hace falta y verifica
async def correr(http: httpx.AsyncClient, escenario: Escenario, timeout_s: float, pausa_s: float) -> Resultado:
    inicio = time.monotonic()
    cuerpo = {"pregunta": escenario.pregunta, "notificar_a": escenario.notificar_a}
    creada = await http.post("/tasks", json=cuerpo)
    if creada.status_code != 202:
        return Resultado(escenario, "-", 0.0, [f"POST /tasks devolvió {creada.status_code}: {creada.text}"])
    job_id: str = creada.json()["job_id"]
    problemas: list[str] = []
    if escenario.notificar_a:
        # Tarea crítica: tiene que frenarse esperando a una persona
        pausado = await esperar(http, job_id, FINALES | {"WAITING_APPROVAL"}, timeout_s, pausa_s)
        if pausado["status"] != "WAITING_APPROVAL":
            problemas.append(f"debía pausarse en WAITING_APPROVAL y está en {pausado['status']}: {pausado.get('error')}")
        elif not pausado.get("aprobacion_pendiente") or pausado.get("despacho"):
            problemas.append("en la pausa debía mostrar qué se aprueba y no haber enviado nada todavía")
        else:
            print(f"      pausado. Se le pide aprobar a un humano: {pausado['aprobacion_pendiente'].get('motivo')}")
            aprobada = await http.post(f"/tasks/{job_id}/approve", json={"aprobado": True, "comentario": "aprobado por pruebas_sistema"})
            if aprobada.status_code != 202:
                problemas.append(f"POST /approve devolvió {aprobada.status_code}: {aprobada.text}")
    final = await esperar(http, job_id, FINALES, timeout_s, pausa_s)
    if final["status"] not in FINALES:
        problemas.append(f"no terminó en {timeout_s:.0f} s (estado: {final['status']})")
    else:
        problemas.extend(escenario.verificar(final))
        print(f"      respuesta: {(final.get('resultado') or final.get('error') or '').strip()[:300]}")
    return Resultado(escenario, job_id, time.monotonic() - inicio, problemas)


# Chequeos previos del contrato HTTP (no usan el LLM): la API está viva y rechaza lo inválido
async def chequeos_previos(http: httpx.AsyncClient) -> list[str]:
    problemas: list[str] = []
    salud = await http.get("/health")
    if salud.status_code != 200:
        problemas.append(f"GET /health devolvió {salud.status_code}: {salud.text}")
    if (await http.post("/tasks", json={"pregunta": "a"})).status_code != 422:
        problemas.append("POST /tasks con una pregunta de 1 letra debería devolver 422")
    if (await http.get("/tasks/no-existe")).status_code != 404:
        problemas.append("GET /tasks/no-existe debería devolver 404")
    return problemas


# Corre los 5 escenarios en orden y devuelve los resultados
async def ejecutar(http: httpx.AsyncClient, timeout_s: float = 180, pausa_s: float = 1.0) -> tuple[list[str], list[Resultado]]:
    previos = await chequeos_previos(http)
    resultados: list[Resultado] = []
    for escenario in ESCENARIOS:
        print(f"\n[{escenario.nombre}]\n      {escenario.objetivo}")
        resultado = await correr(http, escenario, timeout_s, pausa_s)
        print(f"      {'OK' if not resultado.problemas else 'FALLÓ'} en {resultado.segundos:.1f} s · job_id (session.id en Phoenix): {resultado.job_id}")
        resultados.append(resultado)
    return previos, resultados


# Imprime el resumen final y devuelve cuántos problemas hubo
def resumir(previos: list[str], resultados: list[Resultado]) -> int:
    print("\n" + "=" * 78 + "\nRESUMEN\n" + "=" * 78)
    for problema in previos:
        print(f"  FALLÓ chequeo previo: {problema}")
    for resultado in resultados:
        print(f"  {'OK    ' if not resultado.problemas else 'FALLÓ '}{resultado.escenario.nombre}  ({resultado.job_id})")
        for problema in resultado.problemas:
            print(f"        - {problema}")
    total = len(previos) + sum(len(r.problemas) for r in resultados)
    aprobados = sum(1 for r in resultados if not r.problemas)
    print(f"\n{aprobados} de {len(resultados)} escenarios OK. Las trazas están en Phoenix (http://localhost:6006): buscá cada job_id en session.id.")
    return total


# Punto de entrada: lee los parámetros, corre las pruebas y devuelve el código de salida
async def main() -> int:
    parser = argparse.ArgumentParser(description="Pruebas del sistema contra la API levantada")
    parser.add_argument("--url", default="http://localhost:8000", help="URL base de la API")
    parser.add_argument("--timeout", type=float, default=180, help="segundos máximos de espera por escenario")
    argumentos = parser.parse_args()
    async with httpx.AsyncClient(base_url=argumentos.url, timeout=30) as http:
        try:
            previos, resultados = await ejecutar(http, argumentos.timeout)
        except httpx.ConnectError:
            print(f"No se pudo conectar con {argumentos.url}. ¿Está levantado el sistema? (ver README: ./run.sh)")
            return 2
    return 1 if resumir(previos, resultados) else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
