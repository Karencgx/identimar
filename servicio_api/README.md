# IdentiMar — Servicio de visión artificial (API REST)

Servicio que recibe una foto de una raya desde el **backend (Equipo 3)** y devuelve la especie propuesta, la confianza y un estado
(`identificado` / `no_concluyente`), usando el modelo YOLOv8n congelado. También recibe las **observaciones validadas por expertos**
para realimentar el dataset.

```
App móvil (Equipo 1) → Backend (Equipo 3) → POST /predict → este servicio → JSON → Backend → App móvil
                                           → POST /feedback (cuando un experto confirma o corrige)
```

El Equipo 1 nunca llama a este servicio: solo ve el resultado normalizado que le reenvía el backend.

## Contenido

```
app/            código del servicio (main.py = endpoints; contrato_salida.py = salida estándar)
tests/          63 pruebas automáticas con un detector simulado
scripts/        probar_api.py: prueba de integración + medición de latencia
models/         aquí se coloca best.pt (no se sube al repositorio)
Dockerfile, docker-compose.yml, requirements.txt, openapi.json
```

## Puesta en marcha con Docker

Requiere Docker en su computador o en el servidor (Docker **no** funciona dentro de Colab).

1. Copien el modelo congelado: `modelo_final_v1/pesos/best.pt` → `models/best.pt`.
2. Creen el archivo `.env` (copien `.env.example`) y cambien `API_KEY`.
3. Levanten el servicio:

```bash
docker compose up --build -d
docker compose logs -f          # debe aparecer "Modelo cargado: /models/best.pt (sha256 ...)"
curl http://localhost:8000/health
```

La primera construcción tarda varios minutos (descarga PyTorch CPU). La imagen no incluye el modelo: se monta desde `models/`,
así pueden cambiar de modelo sin reconstruir. La documentación interactiva queda en `http://localhost:8000/docs`.

## Probarlo sin Docker (por ejemplo, en Colab)

```python
!unzip -q identimar_api.zip -d /content
!pip install -q fastapi uvicorn python-multipart httpx      # ultralytics ya está instalado

import os, subprocess
os.environ["MODEL_PATH"] = "/content/modelo_final_v1/pesos/best.pt"
os.environ["FEEDBACK_DIR"] = "/content/feedback_api"
servidor = subprocess.Popen(["uvicorn", "app.main:app", "--port", "8000"], cwd="/content/identimar_api")
```

Luego, en otra celda: `!curl localhost:8000/health` y `!python /content/identimar_api/scripts/probar_api.py --carpeta /content/IdentiMar-Rayas-2/externas`.

## Endpoints

| Método y ruta | Para qué sirve |
|---|---|
| `GET /health` | Estado del servicio y del modelo (200 listo, 503 si el modelo no cargó). No exige API key. |
| `POST /predict` | Identificar una imagen enviada como archivo (`multipart/form-data`). |
| `POST /predict/url` | Igual, pero con una URL https de un almacenamiento permitido. **Deshabilitado por defecto.** |
| `POST /feedback` | Ingresar una observación validada por un experto. |
| `POST /predict/visual` | Igual que `/predict`, pero devuelve un JPEG con las detecciones dibujadas (resultado del contrato en el encabezado `X-Resultado`). |
| `GET /especies` | Catálogo de especies del modelo (id, clase, nombre científico, nombre común) para armonizar la taxonomía. |
| `GET /demo` | Página web de demostración: subir una foto y ver detecciones, especie, confianza y tiempos. |
| `GET /feedback/export` | Descargar todo el feedback en un zip (respaldo si el disco del servidor no es persistente). |

### `POST /predict`

```bash
curl -i -X POST http://localhost:8000/predict \
  -H "X-API-Key: $API_KEY" \
  -F id_observacion=obs-0001 \
  -F imagen=@foto.jpg
```

```json
{"id_observacion": "obs-0001", "especie_propuesta": "Hypanus_guttatus", "confianza": 0.9132,
 "umbral_minimo": 0.7, "estado": "identificado",
 "version_modelo": "yolov8n-identimar-1.0.0+3fa9c01b22de", "tiempo_inferencia_ms": 142.3}
```

- Si la confianza es menor a `umbral_minimo` → `estado = "no_concluyente"` (el backend deriva a revisión experta).
- Si el modelo no detecta ninguna raya → `especie_propuesta = null`, `confianza = 0.0`, `estado = "no_concluyente"`.
- El tiempo de inferencia viaja en el encabezado `X-Tiempo-Inferencia-ms` y la versión del modelo en `X-Modelo-Version`
  (los 12 primeros caracteres del sha256 de `best.pt`). Los 5 campos del contrato original se conservan; la versión 1.1 agrega `version_modelo` y `tiempo_inferencia_ms`.
- Se acepta JPEG, PNG y WEBP; la orientación EXIF se aplica antes de inferir.

### `POST /feedback`

```bash
curl -X POST http://localhost:8000/feedback -H "X-API-Key: $API_KEY" \
  -F id_observacion=obs-0001 -F especie_confirmada=Hypanus_guttatus -F validado_por="Dra. Ejemplo" -F imagen=@foto.jpg
```

Guarda, dentro del volumen `feedback_data` (`/data/feedback`):

```
images/<sha256>.jpg     la imagen
labels/<sha256>.txt     etiqueta YOLO: clase confirmada por el experto + caja del modelo
metadata.jsonl          una línea por observación (experto, fecha, predicción original, versión del modelo)
```

**La caja la propone el propio modelo**, por eso cada registro lleva `requiere_revision_caja = true`: revisen las cajas
(por ejemplo subiendo `images/` y `labels/` a Roboflow) antes de reentrenar. Si el modelo no detectó nada, no se crea etiqueta y el
registro queda con `requiere_anotacion_manual = true`. Una imagen repetida responde `409`.

Para sacar los datos del volumen: `docker volume ls` (busquen `..._feedback_data`) y luego
`docker run --rm -v NOMBRE_DEL_VOLUMEN:/data -v "$PWD":/out alpine tar czf /out/feedback.tgz -C /data .`

## Despliegue gratuito en Google Cloud Run

Cloud Run tiene una cuota gratuita permanente y se apaga solo cuando no hay uso. Requiere una cuenta de Google con facturación activada (tarjeta), sin cobro mientras se respete la cuota.


```bash
gcloud run deploy identimar-vision --source . --region us-central1 --port 8000 \
  --memory 2Gi --cpu 1 --timeout 120 --concurrency 4 --min-instances 0 --max-instances 2 \
  --allow-unauthenticated --env-vars-file ~/env.yaml        # ver cloudrun/env.yaml.example
```

No active "instancias mínimas": mantener una instancia encendida consume la cuota gratuita.

## Códigos de error

| Código | Cuándo |
|---|---|
| 400 | Imagen inválida, corrupta, vacía, muy pequeña (< 32 px) o de formato no soportado; `id_observacion` o especie inválidos |
| 401 | Falta la API key o es incorrecta |
| 403 / 501 / 502 | `/predict/url`: host no permitido / función deshabilitada / el almacenamiento no respondió bien |
| 409 | `/feedback`: la imagen ya había sido ingresada |
| 413 | Imagen de más de `MAX_UPLOAD_MB` MB o de más de `MAX_PIXELS` píxeles |
| 422 | Faltan campos obligatorios |
| 500 | Falló la inferencia (el detalle queda en el log, no en la respuesta) |
| 503 | El modelo no está cargado |

## Variables de entorno

| Variable | Por defecto | Descripción |
|---|---|---|
| `MODEL_PATH` | `/models/best.pt` | Ruta de los pesos |
| `UMBRAL_CONFIANZA` | `0.70` | Umbral del contrato para `no_concluyente` |
| `CONF_PREDICT` | `0.25` | Piso de confianza del modelo (permite informar la especie sospechada en casos no concluyentes) |
| `API_KEY` | vacía | Si se define, se exige en el encabezado `X-API-Key` |
| `MODEL_VERSION` | `yolov8n-identimar-1.0.0` | Nombre de la versión del modelo; en la respuesta se le agrega el hash corto de `best.pt` |
| `MAX_UPLOAD_MB` / `MAX_PIXELS` | `10` / `50000000` | Límites de tamaño |
| `FEEDBACK_DIR` | `/data/feedback` | Dónde se guarda el feedback |
| `ALLOWED_IMAGE_HOSTS` | vacía | Hosts permitidos para `/predict/url` (separados por coma); vacía = deshabilitado |

## Referencias a almacenamiento (Supabase)

Si el backend guarda las fotos en Supabase Storage, puede enviar una URL firmada a `POST /predict/url`
(`{"id_observacion": "...", "url_imagen": "https://..."}`). Hay que permitir ese host: `ALLOWED_IMAGE_HOSTS=<proyecto>.supabase.co`.

## Scripts de apoyo (`scripts/`)

| Script | Para qué |
|---|---|
| `descargar_candidatas_inat.py` | Descarga fotos nuevas de iNaturalist (licencia abierta, calidad research) con su procedencia, para armar el conjunto ciego. |
| `verificar_duplicados.py` | Comprueba que fotos nuevas no estén ya en el dataset (copia exacta o variante casi idéntica, incluso volteada). |
| `evaluar_externas.py` | Evalúa el modelo con el contrato sobre fotos nuevas organizadas por grupo (`en_agua`, `fuera_del_agua`, ...) y especie; guarda CSV e imágenes anotadas. |
| `ingerir_validadas.py` | Convierte observaciones validadas por expertos (CSV + imágenes) al formato YOLO listo para revisar y reentrenar manualmente. |
| `probar_api.py` | Prueba de integración y medición de latencia contra el servicio desplegado. |

## Pruebas

```bash
pip install -r requirements-dev.txt     # o: pip install pytest httpx fastapi uvicorn python-multipart pillow
pytest -q                                # 63 pruebas, no necesitan el modelo ni GPU
python scripts/probar_api.py --url http://localhost:8000 --carpeta ./fotos_externas --api-key TU_CLAVE
```

`probar_api.py` imprime la respuesta de cada imagen, la latencia (ida y vuelta y de inferencia, con mediana y p95) y guarda
`resultados_api.csv`: sirve de evidencia para la prueba de integración.

## Consideraciones de integración con el backend

1. **Envío de imágenes:** el servicio recibe una URL firmada temporal de Supabase Storage mediante `POST /predict/url`. La URL tiene una duración de 10 minutos.
2. **Errores del servicio:** el backend contempla el código HTTP `503 SERVICIO_VISION_NO_DISPONIBLE` cuando el modelo de visión no responde.
3. **Autenticación y seguridad:** las solicitudes del backend al servicio de visión utilizan el encabezado `x-api-key` con una clave secreta compartida entre los equipos 2 y 3. Todos los servicios utilizan HTTPS.
4. **Prevención de *cold starts*:** UptimeRobot realiza solicitudes periódicas cada 5 minutos al backend y al modelo de visión para evitar tiempos de arranque en frío en Render y Google Cloud Run.
5. **Nombres de especies en `/feedback`:** el campo `especie_confirmada` utiliza los identificadores del catálogo taxonómico unificado, compartidos entre el backend y el modelo de visión.

## Limitaciones conocidas

- Un solo worker; las inferencias se atienden una a la vez. Suficiente para el MVP, no para alta concurrencia.
- Sin HTTPS ni límite de peticiones: debe ir detrás de la red privada del backend o de un proxy inverso.
- La latencia en CPU será mayor que la medida en GPU (Tesla T4) durante el entrenamiento: mídanla en el servidor real con `probar_api.py`.
- El modelo se entrenó con Ultralytics 8.4.170; `requirements.txt` fija esa versión.
