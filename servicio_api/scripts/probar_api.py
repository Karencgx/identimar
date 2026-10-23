"""Prueba de integración sencilla: envía las imágenes de una carpeta a /predict y mide los tiempos.

    pip install requests
    python scripts/probar_api.py --url http://localhost:8000 --carpeta ./fotos_externas --api-key TU_CLAVE

Imprime la respuesta de cada imagen y un resumen de latencia (ida y vuelta completa y tiempo de inferencia
reportado por el servicio). Guarda todo en resultados_api.csv: sirve como evidencia de la prueba de integración.
"""
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import requests

EXT = {".jpg", ".jpeg", ".png", ".webp"}


def percentil(valores, p):
    valores = sorted(valores)
    return valores[min(len(valores) - 1, round(p / 100 * (len(valores) - 1)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--carpeta", required=True)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--salida", default="resultados_api.csv")
    args = ap.parse_args()

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    try:
        salud = requests.get(f"{args.url}/health", timeout=10)
    except requests.ConnectionError:
        sys.exit(f"No se pudo conectar con {args.url}. ¿Está corriendo el servicio?")
    print("health:", salud.status_code, salud.json().get("estado") or salud.json().get("detail", {}).get("estado"))

    fotos = sorted(p for p in Path(args.carpeta).iterdir() if p.suffix.lower() in EXT)
    if not fotos:
        sys.exit(f"No hay imágenes en {args.carpeta}")

    filas, ida_vuelta, inferencia = [], [], []
    for foto in fotos:
        t0 = time.perf_counter()
        with open(foto, "rb") as f:
            r = requests.post(f"{args.url}/predict", headers=headers, data={"id_observacion": foto.stem},
                              files={"imagen": (foto.name, f)}, timeout=60)
        ms_total = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            j = r.json()
            ms_inf = float(r.headers.get("X-Tiempo-Inferencia-ms", "nan"))
            ida_vuelta.append(ms_total)
            inferencia.append(ms_inf)
            print(f"{foto.name:40s} {j['estado']:15s} {str(j['especie_propuesta']):26s} conf={j['confianza']:.3f}  "
                  f"total={ms_total:6.0f} ms  inferencia={ms_inf:6.1f} ms")
            filas.append([foto.name, j["estado"], j["especie_propuesta"], j["confianza"], round(ms_total, 1), ms_inf])
        else:
            print(f"{foto.name:40s} ERROR {r.status_code}: {r.text[:120]}")
            filas.append([foto.name, f"error_{r.status_code}", "", "", round(ms_total, 1), ""])

    with open(args.salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["imagen", "estado", "especie_propuesta", "confianza", "ms_ida_y_vuelta", "ms_inferencia"])
        w.writerows(filas)

    if ida_vuelta:
        print(f"\n{len(ida_vuelta)}/{len(fotos)} respuestas correctas -> {args.salida}")
        print(f"Ida y vuelta (ms): mediana={statistics.median(ida_vuelta):.0f}  p95={percentil(ida_vuelta, 95):.0f}  "
              f"máx={max(ida_vuelta):.0f}")
        print(f"Inferencia   (ms): mediana={statistics.median(inferencia):.1f}  p95={percentil(inferencia, 95):.1f}  "
              f"máx={max(inferencia):.1f}")
        print("Nota: la primera petición puede ser más lenta (calentamiento); descártenla si quieren medir régimen estable.")


if __name__ == "__main__":
    main()
