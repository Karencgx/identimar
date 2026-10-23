"""
IdentiMar - Grupo 9 (Visión Artificial)
Cierre del modelo final (YOLOv8n, sobremuestreo solo de la raya del Magdalena, seed=0).

Hace 4 cosas y deja todo en OUT_DIR (y un .zip para descargar):
  1) Auditoría de la composición real del dataset (sobre todo de train, tras el sobremuestreo)
  2) Evaluación sobre test y validación: tabla P / R / F1 / mAP por clase + matriz de confusión
     en conteos y normalizada
  3) Evaluación OPERATIVA con la salida estándar del contrato (umbral 0.70) + barrido de umbrales
  4) Congelar y versionar: pesos, hash, hiperparámetros, versiones de librerías y ficha del modelo

Uso en Colab:
  1. Subir este archivo y contrato_salida.py al panel de archivos (carpeta /content)
  2. Revisar las rutas de la sección CONFIGURACIÓN
  3. !python cierre_modelo_final.py
"""

import csv
import hashlib
import json
import platform
import re
import shutil
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import yaml

from contrato_salida import generar_respuesta

# ----------------------------------------------------------------------------
# CONFIGURACIÓN (ajusten estas rutas si es necesario)
# ----------------------------------------------------------------------------
RUN_DIR = Path("/content/runs/detect/identimar/baseline_n-3")  # carpeta de la corrida FINAL
DATASET_DIR = Path("/content/IdentiMar-Rayas-2")  # dataset exportado de Roboflow
OUT_DIR = Path("/content/modelo_final_v1")  # aquí se deja todo
UMBRAL = 0.70  # umbral del contrato (confianza < UMBRAL -> no_concluyente)
CONF_PREDICT = 0.25  # conf con el que se predice (bajo, para poder informar casos no concluyentes)
CONF_MATRIZ = 0.25  # conf con el que se calcula la matriz de confusión (ver evaluar_split)

EXT_IMG = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
# Nombre de la carpeta en el dataset -> valor de `split` que entiende Ultralytics
SPLITS = {"test": "test", "valid": "val"}


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.3f}".rstrip("0").rstrip(".") or "0"
    return str(v)


def a_markdown(filas):
    cols = list(filas[0].keys())
    lineas = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for f in filas:
        lineas.append("| " + " | ".join(fmt(f[c]) for c in cols) + " |")
    return "\n".join(lineas)


def escribir_tabla(filas, base):
    """Guarda `base`.csv y `base`.md a partir de una lista de dicts."""
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    cols = list(filas[0].keys())
    with open(base.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for fila in filas:
            w.writerow([round(float(fila[c]), 4) if isinstance(fila[c], (float, np.floating)) else fila[c] for c in cols])
    base.with_suffix(".md").write_text(a_markdown(filas) + "\n", encoding="utf-8")


def leer_clases(dataset_dir):
    cfg = yaml.safe_load(open(Path(dataset_dir) / "data.yaml", encoding="utf-8"))
    names = cfg["names"]
    return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)


def clases_de_etiqueta(label_path):
    if not Path(label_path).exists():
        return []
    out = []
    for linea in Path(label_path).read_text().splitlines():
        partes = linea.split()
        if partes:
            out.append(int(float(partes[0])))
    return out


def listar_imagenes(carpeta):
    carpeta = Path(carpeta)
    if not carpeta.exists():
        return []
    return sorted(p for p in carpeta.iterdir() if p.suffix.lower() in EXT_IMG)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# 1) Auditoría de composición del dataset
# ----------------------------------------------------------------------------
def auditar_particion(dataset_dir, nombres, carpeta):
    dataset_dir = Path(dataset_dir)
    imgs = listar_imagenes(dataset_dir / carpeta / "images")
    inst = {n: 0 for n in nombres}
    img_por_clase = {n: 0 for n in nombres}
    lab_dir = dataset_dir / carpeta / "labels"
    for lab in sorted(lab_dir.glob("*.txt")) if lab_dir.exists() else []:
        cls = clases_de_etiqueta(lab)
        for c in cls:
            inst[nombres[c]] += 1
        for c in set(cls):
            img_por_clase[nombres[c]] += 1
    dup = [p.name for p in imgs if "_dup" in p.stem]
    return {
        "particion": carpeta,
        "imagenes": len(imgs),
        "imagenes_por_clase": img_por_clase,
        "instancias_por_clase": inst,
        "copias_dup": len(dup),
        "copias_dup_anidadas": sum(1 for n in dup if n.count("_dup") > 1),
    }


def imprimir_auditoria(aud):
    for a in aud.values():
        total = a["imagenes"] or 1
        print(f"\n[{a['particion']}] {a['imagenes']} imágenes | copias '_dup': {a['copias_dup']} "
              f"(anidadas: {a['copias_dup_anidadas']})")
        for n, k in a["imagenes_por_clase"].items():
            print(f"    {n:26s} {k:5d} imágenes ({100 * k / total:5.1f}%)   {a['instancias_por_clase'][n]:5d} instancias")
        if a["copias_dup_anidadas"]:
            print("    ⚠ Hay copias de copias (_dup..._dup): el sobremuestreo se aplicó más de una vez sobre la misma carpeta.")


# ----------------------------------------------------------------------------
# 2) Evaluación: tabla por clase y matriz de confusión
# ----------------------------------------------------------------------------
def filas_por_clase(metricas, nombres):
    """Una fila por clase + promedio macro. Usa metricas.summary() y, si no existe, metricas.box."""
    filas = []
    try:
        for f in metricas.summary():
            p, r = float(f["Box-P"]), float(f["Box-R"])
            f1 = float(f["Box-F1"]) if "Box-F1" in f else (2 * p * r / (p + r) if p + r else 0.0)
            filas.append({"Clase": f["Class"], "Imágenes": int(f["Images"]), "Instancias": int(f["Instances"]),
                          "Precisión": p, "Recall": r, "F1": f1,
                          "mAP50": float(f["mAP50"]), "mAP50-95": float(f["mAP50-95"])})
    except Exception:
        box = metricas.box
        for i, c in enumerate(box.ap_class_index):
            p, r = float(box.p[i]), float(box.r[i])
            _, _, ap50, ap = box.class_result(i)
            filas.append({"Clase": nombres[int(c)], "Imágenes": "—", "Instancias": "—",
                          "Precisión": p, "Recall": r, "F1": 2 * p * r / (p + r) if p + r else 0.0,
                          "mAP50": float(ap50), "mAP50-95": float(ap)})
    prom = {"Clase": "Promedio (macro)", "Imágenes": "—",
            "Instancias": sum(f["Instancias"] for f in filas if isinstance(f["Instancias"], int))}
    for k in ("Precisión", "Recall", "F1", "mAP50", "mAP50-95"):
        prom[k] = float(np.mean([f[k] for f in filas]))
    return filas + [prom]


def guardar_matriz(matriz, nombres, destino, prefijo="matriz_confusion"):
    """Filas = predicho, columnas = real (como Ultralytics). Normalizada por columna: cada columna suma 1."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    matriz = np.asarray(matriz, dtype=float)
    etiquetas = list(nombres) + ["background"]
    norm = matriz / (matriz.sum(axis=0, keepdims=True) + 1e-9)
    for sufijo, m, f in (("conteos", matriz, "{:.0f}"), ("normalizada", norm, "{:.3f}")):
        with open(destino / f"{prefijo}_{sufijo}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["predicho \\ real"] + etiquetas)
            for i, fila in enumerate(m):
                w.writerow([etiquetas[i]] + [f.format(v) for v in fila])
    return norm


def evaluar_split(modelo, dataset_dir, carpeta, nombres, out_dir):
    dataset_dir, out_dir = Path(dataset_dir), Path(out_dir)
    metricas = modelo.val(data=str(dataset_dir / "data.yaml"), split=SPLITS[carpeta], plots=True,
                          project=str(out_dir / "evaluacion"), name=carpeta, exist_ok=True, verbose=False)
    destino = Path(getattr(metricas, "save_dir", out_dir / "evaluacion" / carpeta))
    filas = filas_por_clase(metricas, nombres)
    escribir_tabla(filas, destino / "tabla_por_clase")

    # Matriz de confusión: segunda pasada con conf=CONF_MATRIZ.
    # Desde Ultralytics 8.4.177 la matriz se calcula con el mismo `conf` de la validación (0.001 por
    # defecto) y se llena de cientos de falsos positivos "background"; en 8.4.170 usaba 0.25.
    # Las métricas (P/R/F1/mAP) SÍ deben salir de la pasada con conf por defecto; solo la matriz sale de esta.
    met_matriz = modelo.val(data=str(dataset_dir / "data.yaml"), split=SPLITS[carpeta], plots=True, conf=CONF_MATRIZ,
                            project=str(out_dir / "evaluacion"), name=f"{carpeta}_matriz_conf{CONF_MATRIZ}",
                            exist_ok=True, verbose=False)
    dest_m = Path(getattr(met_matriz, "save_dir", out_dir / "evaluacion" / f"{carpeta}_matriz_conf{CONF_MATRIZ}"))
    guardar_matriz(met_matriz.confusion_matrix.matrix, nombres, dest_m)
    try:  # imagen normalizada (Ultralytics ya la guarda sola; esto garantiza que exista)
        met_matriz.confusion_matrix.plot(normalize=True, save_dir=str(dest_m))
    except Exception as e:
        print(f"  (no se pudo redibujar la matriz normalizada: {e})")
    print(f"  Matriz de confusión (conf={CONF_MATRIZ}) guardada en {dest_m}")
    print(f"\n=== {carpeta.upper()} ===\n{a_markdown(filas)}")
    print("Nota: P, R y F1 de Ultralytics se calculan al umbral de confianza que maximiza el F1 promedio, "
          "no al umbral 0.70 del contrato (eso lo mide la evaluación operativa).")
    return filas


# ----------------------------------------------------------------------------
# 3) Evaluación operativa con la salida estándar del contrato
# ----------------------------------------------------------------------------
def barrido_umbral(detalle, umbrales=(0.5, 0.6, 0.7, 0.8, 0.9)):
    n = len(detalle)
    filas = []
    for u in umbrales:
        ident = [d for d in detalle if d["propuesta"] is not None and d["confianza"] >= u]
        ok = sum(1 for d in ident if d["acierta"])
        filas.append({
            "Umbral": u,
            "Identificadas": len(ident),
            "% identificadas": round(100 * len(ident) / n, 1) if n else 0.0,
            "Correctas": ok,
            "% correctas entre identificadas": round(100 * ok / len(ident), 1) if ident else "—",
            "Errores con alta confianza": len(ident) - ok,
            "Derivadas a experto": n - len(ident),
        })
    return filas


def evaluacion_operativa(modelo, dataset_dir, nombres, carpeta, out_dir, umbral=UMBRAL, conf=CONF_PREDICT):
    dataset_dir, out_dir = Path(dataset_dir), Path(out_dir)
    detalle, ejemplos = [], {}
    for img in listar_imagenes(dataset_dir / carpeta / "images"):
        reales = {nombres[c] for c in clases_de_etiqueta(dataset_dir / carpeta / "labels" / f"{img.stem}.txt")}
        resultado = modelo.predict(str(img), conf=conf, verbose=False)[0]
        resp = generar_respuesta(resultado, img.stem, umbral)
        acierta = resp["especie_propuesta"] in reales
        if resp["estado"] == "identificado":
            caso = "identificado_correcto" if acierta else "identificado_INCORRECTO"
        elif resp["especie_propuesta"] is None:
            caso = "no_concluyente_sin_deteccion"
        else:
            caso = "no_concluyente_hubiera_acertado" if acierta else "no_concluyente_hubiera_fallado"
        detalle.append({"imagen": img.name, "especie_real": " + ".join(sorted(reales)),
                        "propuesta": resp["especie_propuesta"], "confianza": resp["confianza"],
                        "estado": resp["estado"], "caso": caso, "acierta": acierta})
        ejemplos.setdefault(caso, {"caso": caso, "imagen": img.name, "especie_real": " + ".join(sorted(reales)),
                                   "respuesta": resp})

    destino = out_dir / "evaluacion" / f"operativa_{carpeta}"
    destino.mkdir(parents=True, exist_ok=True)
    with open(destino / "detalle_por_imagen.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = ["imagen", "especie_real", "propuesta", "confianza", "estado", "caso"]
        w.writerow(cols)
        for d in detalle:
            w.writerow([d[c] for c in cols])
    (destino / "ejemplos_salida_contrato.json").write_text(
        json.dumps(list(ejemplos.values()), ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(detalle)
    conteo = {}
    for d in detalle:
        conteo[d["caso"]] = conteo.get(d["caso"], 0) + 1
    ident = sum(v for k, v in conteo.items() if k.startswith("identificado"))
    resumen = [{"Caso": k, "Imágenes": v, "%": round(100 * v / n, 1) if n else 0.0} for k, v in sorted(conteo.items())]
    escribir_tabla(resumen, destino / "resumen_casos")
    barrido = barrido_umbral(detalle)
    escribir_tabla(barrido, destino / "barrido_umbral")

    print(f"\n=== OPERATIVA ({carpeta}, {n} imágenes, umbral {umbral}) ===\n{a_markdown(resumen)}")
    if ident:
        ok = conteo.get("identificado_correcto", 0)
        print(f"\nDe {n} imágenes, {ident} se identifican solas ({100 * ident / n:.0f}%) y {ok} de esas acertaron "
              f"({100 * ok / ident:.0f}%). {n - ident} se derivan a experto.")
    print(f"\nBarrido de umbrales:\n{a_markdown(barrido)}")
    return detalle, resumen, barrido


# ----------------------------------------------------------------------------
# 4) Congelar y versionar el modelo
# ----------------------------------------------------------------------------
CLAVES_HIPERPARAMETROS = ["model", "data", "epochs", "patience", "batch", "imgsz", "seed", "deterministic",
                          "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", "hsv_h",
                          "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "fliplr", "flipud",
                          "mosaic", "mixup", "close_mosaic", "amp", "pretrained"]
ARCHIVOS_ENTRENAMIENTO = ["args.yaml", "results.csv", "results.png", "confusion_matrix.png",
                          "confusion_matrix_normalized.png", "F1_curve.png", "PR_curve.png", "P_curve.png",
                          "R_curve.png"]


def mejor_epoca(results_csv):
    try:
        with open(results_csv, encoding="utf-8") as f:
            filas = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
        fit = lambda r: 0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"])
        return int(float(max(filas, key=fit)["epoch"])), len(filas)
    except Exception:
        return None, None


def versiones_entorno():
    v = {"python": platform.python_version(), "so": platform.platform()}
    for pkg in ("ultralytics", "torch", "roboflow", "numpy"):
        try:
            v[pkg] = version(pkg)
        except PackageNotFoundError:
            v[pkg] = "no instalado"
    return v


def congelar_modelo(run_dir, dataset_dir, out_dir, nombres, auditoria, metricas_finales):
    run_dir, dataset_dir, out_dir = Path(run_dir), Path(dataset_dir), Path(out_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"No existe {best}. Revisen RUN_DIR.")
    (out_dir / "pesos").mkdir(parents=True, exist_ok=True)
    (out_dir / "entrenamiento").mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out_dir / "pesos" / "best.pt")
    for nombre in ARCHIVOS_ENTRENAMIENTO:
        if (run_dir / nombre).exists():
            shutil.copy2(run_dir / nombre, out_dir / "entrenamiento" / nombre)
    shutil.copy2(dataset_dir / "data.yaml", out_dir / "dataset_data.yaml")

    args = {}
    if (run_dir / "args.yaml").exists():
        args = yaml.safe_load(open(run_dir / "args.yaml", encoding="utf-8")) or {}
    hiper = {k: args[k] for k in CLAVES_HIPERPARAMETROS if k in args}
    epoca, n_epocas = mejor_epoca(run_dir / "results.csv")
    m = re.search(r"-(\d+)$", dataset_dir.name)

    ficha = {
        "nombre": "identimar-rayas-yolov8n",
        "fecha_congelado_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "arquitectura": hiper.get("model", "yolov8n.pt"),
        "clases": {i: n for i, n in enumerate(nombres)},
        "pesos": {"archivo": "pesos/best.pt", "sha256": sha256(out_dir / "pesos" / "best.pt"),
                  "tamano_mb": round((out_dir / "pesos" / "best.pt").stat().st_size / 1e6, 2)},
        "entrenamiento": {"corrida": str(run_dir), "mejor_epoca": epoca, "epocas_ejecutadas": n_epocas,
                          "hiperparametros": hiper},
        "dataset": {"carpeta": dataset_dir.name,
                    "version_roboflow_deducida_del_nombre": m.group(1) if m else None,
                    "auditoria": auditoria},
        "contrato": {"umbral_no_concluyente": UMBRAL, "conf_predict": CONF_PREDICT,
                     "campos": ["id_observacion", "especie_propuesta", "confianza", "umbral_minimo", "estado"]},
        "metricas": {k: v for k, v in metricas_finales.items()},
        "entorno": versiones_entorno(),
    }
    (out_dir / "ficha_modelo.json").write_text(json.dumps(ficha, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    md = [f"# Ficha del modelo — {ficha['nombre']}", "",
          f"- **Congelado:** {ficha['fecha_congelado_utc']} UTC",
          f"- **Arquitectura / pesos de partida:** {ficha['arquitectura']} (transferencia de aprendizaje desde COCO)",
          f"- **Pesos:** `pesos/best.pt` · {ficha['pesos']['tamano_mb']} MB · sha256 `{ficha['pesos']['sha256']}`",
          f"- **Mejor época:** {epoca} de {n_epocas} ejecutadas · seed = {hiper.get('seed')}",
          f"- **Dataset:** carpeta `{dataset_dir.name}` (versión de Roboflow deducida del nombre: {ficha['dataset']['version_roboflow_deducida_del_nombre']}; confirmar)",
          f"- **Clases:** " + ", ".join(f"{i}={n}" for i, n in enumerate(nombres)),
          f"- **Contrato:** confianza < {UMBRAL} → `no_concluyente`", "",
          "## Composición del dataset (auditoría)"]
    for a in auditoria.values():
        md.append(f"- **{a['particion']}**: {a['imagenes']} imágenes; por clase: " +
                  ", ".join(f"{n}={k}" for n, k in a["imagenes_por_clase"].items()) +
                  f"; copias `_dup`: {a['copias_dup']}")
    md += ["", "## Hiperparámetros", "```yaml", yaml.safe_dump(hiper, allow_unicode=True).strip(), "```", "",
           "## Métricas finales (P, R y F1 al umbral de máximo F1 de Ultralytics)"]
    for split, filas in metricas_finales.items():
        md += [f"### {split}", a_markdown(filas), ""]
    md += ["## Entorno", "```json", json.dumps(ficha["entorno"], indent=2), "```", "",
           "## Completar a mano",
           "- [ ] Enlace al repositorio y commit/tag de este modelo: ",
           "- [ ] Proyecto y versión de Roboflow (confirmar): ",
           "- [ ] Sobremuestreo aplicado (clase, copias por imagen, imágenes de origen): ",
           "- [ ] Responsable de la corrida: "]
    (out_dir / "FICHA_MODELO.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return shutil.make_archive(str(out_dir), "zip", root_dir=out_dir.parent, base_dir=out_dir.name)


# ----------------------------------------------------------------------------
# Flujo completo
# ----------------------------------------------------------------------------
def ejecutar_todo(modelo, run_dir=RUN_DIR, dataset_dir=DATASET_DIR, out_dir=OUT_DIR):
    run_dir, dataset_dir, out_dir = Path(run_dir), Path(dataset_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nombres = leer_clases(dataset_dir)

    print("##### 1) Auditoría del dataset #####")
    auditoria = {c: auditar_particion(dataset_dir, nombres, c) for c in ("train", "valid", "test")
                 if (dataset_dir / c).exists()}
    imprimir_auditoria(auditoria)

    print("\n##### 2) Evaluación por clase y matrices de confusión #####")
    finales = {c: evaluar_split(modelo, dataset_dir, c, nombres, out_dir) for c in SPLITS}

    print("\n##### 3) Evaluación operativa con el contrato #####")
    evaluacion_operativa(modelo, dataset_dir, nombres, "test", out_dir)

    print("\n##### 4) Congelando el modelo #####")
    zip_path = congelar_modelo(run_dir, dataset_dir, out_dir, nombres, auditoria, finales)
    print(f"\nListo. Todo quedó en {out_dir}\nZIP para descargar/subir al repo: {zip_path}")
    return zip_path


def main():
    from ultralytics import YOLO

    modelo = YOLO(str(RUN_DIR / "weights" / "best.pt"))
    zip_path = ejecutar_todo(modelo)
    try:  # solo funciona dentro de Colab
        from google.colab import files
        files.download(zip_path)
    except Exception:
        pass


if __name__ == "__main__":
    main()
