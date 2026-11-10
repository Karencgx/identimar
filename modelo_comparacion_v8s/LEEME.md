# Comparación con YOLOv8s

Corrida `comparacion_s-2`, hecha con el mismo procedimiento del modelo definitivo (V7, YOLOv8n):
dataset Roboflow versión 2, sobremuestreo idempotente de *P. magdalenae* (x3, solo train), `seed=0`,
100 épocas máximo, `patience=20`, `batch=16`, `imgsz=640`, Ultralytics 8.4.170.
Entrenó 74 épocas (mejor época: 54, parada temprana) en 0.333 h con una GPU Tesla T4.

## Resultado frente al V7 (YOLOv8n)

| Modelo | Prueba mAP50 | Prueba mAP50-95 | Validación mAP50 | Validación mAP50-95 | Pesos |
|---|---|---|---|---|---|
| YOLOv8n (V7, definitivo) | 0.950 | 0.759 | 0.834 | 0.598 | 6.3 MB |
| YOLOv8s | 0.906 | 0.722 | 0.798 | 0.570 | 22.5 MB |

Una sola semilla y 29 imágenes de prueba: la diferencia no basta para afirmar que YOLOv8n sea mejor en general,
pero no hay evidencia a favor del modelo mayor, que además pesa unas 3.6 veces más.

## Prueba (29 imágenes)

| Clase | Precisión | Recall | F1* | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| Aetobatus_narinari | 0.849 | 0.875 | 0.862 | 0.859 | 0.679 |
| Hypanus_guttatus | 0.789 | 1.000 | 0.882 | 0.995 | 0.709 |
| Potamotrygon_magdalenae | 1.000 | 0.571 | 0.727 | 0.881 | 0.691 |
| Styracura_schmardae | 0.973 | 0.750 | 0.847 | 0.889 | 0.808 |
| **Todas** | 0.903 | 0.799 | 0.848 | 0.906 | 0.722 |

## Validación (58 imágenes)

| Clase | Precisión | Recall | F1* | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| Aetobatus_narinari | 0.916 | 1.000 | 0.956 | 0.988 | 0.707 |
| Hypanus_guttatus | 0.543 | 1.000 | 0.704 | 0.789 | 0.504 |
| Potamotrygon_magdalenae | 0.516 | 0.556 | 0.535 | 0.576 | 0.461 |
| Styracura_schmardae | 0.854 | 0.750 | 0.799 | 0.841 | 0.610 |
| **Todas** | 0.707 | 0.826 | 0.762 | 0.798 | 0.570 |

\* F1 calculado a partir de precisión y recall de la tabla de Ultralytics; no es el F1 en el umbral que lo maximiza.

## Archivos

- `pesos/best.pt`: sha256 `cd7b65dab2208a2c625565dd0def30d3e24bd66714b0582a813546de0ac5d520`
- `entrenamiento/`: `args.yaml`, `results.csv`, `results.png` y matrices de confusión del entrenamiento
- `evaluacion_test/`: curvas y matrices de la evaluación en prueba
