"""Evalúa el modelo con el contrato (umbral 0.70) sobre un conjunto de imágenes NUEVAS organizado por grupo y especie.

Estructura esperada (los nombres de las carpetas de especie deben ser los de las clases del modelo):

    fotos_ciegas/
        en_agua/Hypanus_guttatus/foto1.jpg
        fuera_del_agua/Potamotrygon_magdalenae/foto2.jpg
        calidad_baja/Aetobatus_narinari/foto3.jpg

    python scripts/evaluar_externas.py --modelo models/best.pt --carpeta ./fotos_ciegas --salida ./resultados_ciegas

Entrega, por grupo y por especie: identificadas, correctas, no concluyentes, errores con alta confianza y tiempos;
guarda el detalle en CSV y las imágenes con las detecciones dibujadas (para el informe y la demostración).
Importante para que la evaluación sea honesta: córranla UNA vez con el modelo definitivo y no elijan el modelo mirando
estos resultados.
"""
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image  # noqa: E402

from app.contrato_salida import generar_respuesta  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".webp"}


def _norm(s):
    return s.strip().replace(" ", "_").lower()


def evaluar(modelo, carpeta, salida=None, umbral=0.70, conf=0.25):
    carpeta = Path(carpeta)
    nombres = {_norm(n): n for n in modelo.names.values()} if hasattr(modelo, "names") else {}
    filas = []
    for grupo in sorted(d for d in carpeta.iterdir() if d.is_dir()):
        for esp in sorted(d for d in grupo.iterdir() if d.is_dir()):
            real = nombres.get(_norm(esp.name))
            if real is None:
                print(f"AVISO: la carpeta '{esp.name}' no es una clase del modelo; se omite.")
                continue
            for foto in sorted(p for p in esp.iterdir() if p.suffix.lower() in EXT):
                t0 = time.perf_counter()
                res = modelo.predict(str(foto), conf=conf, verbose=False)[0]
                ms = (time.perf_counter() - t0) * 1000
                r = generar_respuesta(res, foto.stem, umbral)
                ok = r["especie_propuesta"] == real
                if r["estado"] == "identificado":
                    caso = "identificado_correcto" if ok else "identificado_INCORRECTO"
                else:
                    caso = "no_concluyente_hubiera_acertado" if ok else "no_concluyente_hubiera_fallado"
                filas.append({"grupo": grupo.name, "especie_real": real, "imagen": foto.name,
                              "propuesta": r["especie_propuesta"], "confianza": r["confianza"], "estado": r["estado"],
                              "caso": caso, "ms": round(ms, 1)})
                if salida is not None:
                    try:
                        arr = res.plot()  # BGR de Ultralytics
                        dest = Path(salida) / "anotadas" / grupo.name
                        dest.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(arr[..., ::-1]).save(dest / f"{real}__{foto.stem}.jpg", quality=90)
                    except Exception:
                        pass
    return filas


def resumir(filas, clave):
    out = []
    for valor in sorted({f[clave] for f in filas}):
        sub = [f for f in filas if f[clave] == valor]
        ident = [f for f in sub if f["estado"] == "identificado"]
        ok = [f for f in ident if f["caso"] == "identificado_correcto"]
        out.append({clave: valor, "imágenes": len(sub), "identificadas": len(ident), "correctas": len(ok),
                    "errores con alta confianza": len(ident) - len(ok), "no concluyentes": len(sub) - len(ident),
                    "mediana ms": round(statistics.median(f["ms"] for f in sub), 1)})
    return out


def tabla_md(filas):
    if not filas:
        return "(sin datos)"
    cols = list(filas[0])
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(str(f[c]) for c in cols) + " |" for f in filas])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--carpeta", required=True)
    ap.add_argument("--salida", default="resultados_ciegas")
    ap.add_argument("--umbral", type=float, default=0.70)
    a = ap.parse_args()
    from ultralytics import YOLO

    filas = evaluar(YOLO(a.modelo), a.carpeta, a.salida, a.umbral)
    if not filas:
        sys.exit("No se encontraron imágenes. Revise la estructura grupo/especie/foto.jpg")
    Path(a.salida).mkdir(parents=True, exist_ok=True)
    with open(Path(a.salida) / "detalle.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    texto = f"## Por grupo\n{tabla_md(resumir(filas, 'grupo'))}\n\n## Por especie\n{tabla_md(resumir(filas, 'especie_real'))}\n"
    (Path(a.salida) / "resumen.md").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Detalle e imágenes anotadas en {a.salida}")


if __name__ == "__main__":
    main()
