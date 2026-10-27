# Entrenamiento

- `sobremuestreo_idempotente.py`: sobremuestreo de una clase en `train` (no duplica copias aunque se ejecute varias veces).
- `cierre_modelo_final.py`: evalúa en validación y prueba y genera tablas P/R/F1/mAP, matrices de confusión, evaluación con el contrato y la ficha del modelo.
- `contrato_salida.py`: convierte el resultado del modelo en la respuesta del contrato (la usa `cierre_modelo_final.py`).
- `IdentiMar.ipynb`: notebook de entrenamiento. versión de Ultralytics:8.4.170, versión de proyecto Roboflow:2, semilla:0
