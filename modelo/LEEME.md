# Modelo definitivo (V7)

Copien aquí el contenido de `modelo_final_v1.zip` **excepto** `pesos/best.pt`:

- `FICHA_MODELO.md` y `ficha_modelo.json` (hash de los pesos, hiperparámetros, versiones, métricas)
- `evaluacion/` (tablas por clase, matrices de confusión, evaluación operativa y barrido de umbrales)
- `entrenamiento/` (`args.yaml`, `results.csv`, curvas)
- `dataset_data.yaml`

Los pesos (`best.pt`) van en `servicio_api/models/best.pt`: así hay una sola copia y el servicio se reconstruye con `git clone`.
Comprueben que el sha256 de ese archivo coincide con el de `ficha_modelo.json`.
