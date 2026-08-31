# -----------------------------------------------------------------------------
# ETAPA 1: Construcción de dependencias
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Instalar herramientas de compilación en caso de ser necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Instalar librerías en una carpeta independiente
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# -----------------------------------------------------------------------------
# ETAPA 2: Imagen final de producción
# -----------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Copiar dependencias compiladas desde la etapa anterior
COPY --from=builder /install /usr/local

# Crear un usuario sin privilegios por seguridad
RUN adduser --disabled-password --gecos "" appuser

# Copiar el código de la aplicación y el artefacto del modelo
COPY servicio_api.py .
COPY Salidas/ /app/Salidas/

# Cambiar permisos al usuario no raíz
RUN chown -R appuser:appuser /app
USER appuser

# Exponer el puerto de FastAPI
EXPOSE 8000

# Variables de entorno
ENV PYTHONUNBUFFERED=1

# Comando de inicio con Uvicorn
CMD ["uvicorn", "servicio_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]