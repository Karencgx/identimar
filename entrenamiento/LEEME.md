# Entrenamiento

- `sobremuestreo_idempotente.py`: sobremuestreo de una clase en `train` (no duplica copias aunque se ejecute varias veces).
- `cierre_modelo_final.py`: evalúa en validación y prueba y genera tablas P/R/F1/mAP, matrices de confusión, evaluación con el contrato y la ficha del modelo.
- `contrato_salida.py`: convierte el resultado del modelo en la respuesta del contrato (la usa `cierre_modelo_final.py`).

**Falta añadir aquí:** el notebook de entrenamiento (en Colab: *Archivo → Descargar → Descargar .ipynb*), con las celdas ya ejecutadas si es posible.
Anoten en el notebook o aquí: proyecto y versión de Roboflow, versión de Ultralytics (8.4.170), semilla y configuración de aumentación.
