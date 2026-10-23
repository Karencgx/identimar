FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Librerías del sistema que necesita OpenCV (dependencia de ultralytics)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# 1) PyTorch solo-CPU primero: es la capa más pesada y casi nunca cambia, así se cachea
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2) Resto de dependencias (versiones fijadas en requirements.txt)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3) Código del servicio (el modelo NO va en la imagen: se monta en /models)
COPY app ./app

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/feedback /models \
    && chown -R appuser /data /models

# El modelo viaja DENTRO de la imagen (privada) si existe models/best.pt al construir.
# En docker compose, el volumen ./models:/models lo reemplaza; si no hay modelo aquí ni en un volumen,
# se puede descargar al arrancar desde un repositorio privado (variables HF_*).
COPY --chown=appuser models/ /models/
USER appuser

ENV MODEL_PATH=/models/best.pt \
    FEEDBACK_DIR=/data/feedback \
    UMBRAL_CONFIANZA=0.70 \
    CONF_PREDICT=0.25

EXPOSE 8000

# Healthy solo cuando el modelo está cargado (/health responde 503 si no lo está)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# Un solo worker: el modelo se carga una vez y las inferencias se serializan dentro del proceso
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
