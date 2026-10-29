# IdentiMar — Módulo de visión artificial (Equipo 2)

Reconocimiento de cuatro especies de rayas del Urabá antioqueño con YOLOv8, empaquetado como servicio REST.
Proyecto Integrador I · Ingeniería de Sistemas · Universidad de Antioquia · Grupo 9.

**Equipo:** Andrés Camilo Areiza Londoño · Karen Cardona Gutiérrez · Juan Pablo Valencia Martínez · Sebastián Flórez Jaramillo 

**Profesor:** David S. Fernández Mc Cann

## Qué hay en este repositorio

| Carpeta | Contenido |
|---|---|
| `servicio_api/` | API REST (FastAPI + Docker): `/predict`, `/especies`, `/health`, `/demo`, `/feedback`. Incluye pruebas, scripts de apoyo y la configuración de despliegue en Google Cloud Run. Ver su `README.md`. |
| `entrenamiento/` | Scripts del cierre del modelo (métricas, matriz de confusión, ficha y versionado) y del sobremuestreo. |
| `modelo/` | Ficha y resultados del modelo definitivo (V7): métricas, matrices, curvas e hiperparámetros. |
| `docs/` | Contrato de integración, informe de avance, inventario del dataset y diagramas. |

## Modelo definitivo (V7)

YOLOv8n por transferencia de aprendizaje (COCO), sobremuestreo 3× de la clase *Potamotrygon magdalenae*, Ultralytics 8.4.170.
mAP50 / mAP50-95: **0.950 / 0.759** en prueba (29 imágenes) y **0.834 / 0.598** en validación (58 imágenes).
Con el umbral del contrato (0.70) se identifican 22 de 29 imágenes de prueba y 21 de ellas son correctas.
La conciliación con las demás versiones está en el informe (sección 6.5), dentro de `docs/`.

Especies: *Hypanus guttatus*, *Aetobatus narinari*, *Styracura schmardae*, *Potamotrygon magdalenae*.

## Cómo ejecutar el servicio

```bash
cd servicio_api
cp .env.example .env            # y cambie API_KEY
docker compose up --build -d
curl http://localhost:8000/health
```

Pruebas automáticas (sin modelo ni GPU): `cd servicio_api && pip install -r requirements-dev.txt && pytest -q`.
Despliegue en Cloud Run y variables de entorno: ver `servicio_api/README.md`.

## Cómo reproducir el entrenamiento

1. Dataset: proyecto de Roboflow `IdentiMar-Rayas`, versión 2. Las imágenes no se incluyen por sus licencias; el inventario con procedencia está en `docs/`.
2. Notebook: `entrenamiento/` (exportado desde Colab). Fijar `pip install ultralytics==8.4.170`.
3. Sobremuestreo: `entrenamiento/sobremuestreo_idempotente.py` (3× solo para *P. magdalenae*, solo en `train`).
4. Evaluación y congelado del modelo: `entrenamiento/cierre_modelo_final.py`.

## Seguridad

No se versionan claves ni archivos `.env`. La API exige el encabezado `X-API-Key`; la clave se entrega por un canal seguro.
