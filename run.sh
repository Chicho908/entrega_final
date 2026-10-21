#!/usr/bin/env bash
# Arranque en una sola instruccion (Linux, macOS o Git Bash en Windows):
#   ./run.sh            levanta todo el sistema (equivale a: docker compose up --build)
#   ./run.sh pruebas    corre las 5 pruebas del sistema contra el sistema ya levantado
set -euo pipefail
cd "$(dirname "$0")"

# Sin .env no hay API keys: se crea desde el ejemplo y se corta con instrucciones claras
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Se creo .env desde .env.example. Completa tus API keys (LLM y Pinecone) y volve a ejecutar ./run.sh"
  exit 1
fi

if [ "${1:-}" = "pruebas" ]; then
  docker compose exec api python -m scripts.pruebas_sistema --url http://localhost:8000
else
  docker compose up --build
fi
