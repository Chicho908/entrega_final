# Imagen de la API, del worker y de la ingesta (el mismo codigo; cambia el comando de arranque)
FROM python:3.12-slim

# Logs sin buffer (se ven al instante en docker compose logs) y sin archivos .pyc
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /srv

# Primero las dependencias: si solo cambia el codigo, Docker reutiliza esta capa
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Despues el codigo, los PDF de las politicas (el corpus) y los scripts
COPY app ./app
COPY data ./data
COPY scripts ./scripts

EXPOSE 8000

# Por defecto arranca la API; el compose lo reemplaza por: python -m app.worker  /  python -m app.rag.ingesta
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
