"""Incorpora al dataset imágenes con identificación experta VALIDADA, sin reentrenar nada (proceso manual y documentado).

Entrada: un CSV con las columnas  id_observacion, archivo, especie_confirmada, validado_por  (esta última es opcional).
Pueden salir de la exportación de observaciones validadas del backend.

    python scripts/ingerir_validadas.py --csv validadas.csv --base ./imagenes_validadas --salida ./nuevas_validadas \
                                        [--modelo models/best.pt]

Resultado (igual que /feedback, y listo para revisar en Roboflow antes de cualquier reentrenamiento):
    <salida>/images/<sha256>.jpg   <salida>/labels/<sha256>.txt   <salida>/metadata.jsonl

La clase la pone el experto. La caja se propone con el modelo (si se da --modelo) y SIEMPRE debe revisarse;
sin modelo, la observación queda marcada para anotación manual. Una imagen repetida se omite.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.contrato_salida import generar_respuesta  # noqa: E402
from app.detector import mejor_caja  # noqa: E402
from app.feedback import FeedbackDuplicado, guardar_feedback  # noqa: E402
from app.imagenes import decodificar_imagen  # noqa: E402

CLASES_POR_DEFECTO = ["Aetobatus_narinari", "Hypanus_guttatus", "Potamotrygon_magdalenae", "Styracura_schmardae"]


def ingerir(filas, base, salida, modelo=None, nombres=None, version_modelo=None):
    nombres = nombres or (list(modelo.names.values()) if modelo is not None else CLASES_POR_DEFECTO)
    mapa = {n.lower(): (i, n) for i, n in enumerate(nombres)}
    resumen = {"almacenadas": 0, "duplicadas": 0, "errores": []}
    for f in filas:
        try:
            clave = f["especie_confirmada"].strip().replace(" ", "_").lower()
            if clave not in mapa:
                raise ValueError(f"especie desconocida: {f['especie_confirmada']}")
            class_id, especie = mapa[clave]
            datos = (Path(base) / f["archivo"]).read_bytes()
            img = decodificar_imagen(datos, 50_000_000)
            caja = pred = None
            if modelo is not None:
                res = modelo.predict(img, conf=0.25, verbose=False)[0]
                caja, pred = mejor_caja(res), generar_respuesta(res, f["id_observacion"], 0.70)
            guardar_feedback(salida, img, datos, f["id_observacion"], especie, class_id, pred, caja, version_modelo,
                             f.get("validado_por", ""))
            resumen["almacenadas"] += 1
        except FeedbackDuplicado:
            resumen["duplicadas"] += 1
        except Exception as e:
            resumen["errores"].append(f"{f.get('id_observacion', '?')}: {e}")
    return resumen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--base", default=".", help="carpeta a la que son relativas las rutas de la columna 'archivo'")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--modelo", help="best.pt, para proponer la caja (opcional)")
    a = ap.parse_args()
    modelo = None
    if a.modelo:
        from ultralytics import YOLO
        modelo = YOLO(a.modelo)
    with open(a.csv, newline="", encoding="utf-8-sig") as fh:
        filas = list(csv.DictReader(fh))
    r = ingerir(filas, a.base, a.salida, modelo, version_modelo=a.modelo)
    print(f"Almacenadas: {r['almacenadas']} · duplicadas: {r['duplicadas']} · con error: {len(r['errores'])}")
    for e in r["errores"]:
        print("  ERROR", e)


if __name__ == "__main__":
    main()
