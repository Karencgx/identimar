"""
IdentiMar - Grupo 9 (Visión Artificial)
Sobremuestreo de UNA clase en la carpeta train, versión idempotente.

Por qué existe: la primera versión del script recorría TODOS los .txt de train/labels, incluidos
los *_dup1/2/3 que había creado en una ejecución anterior. Si se corría más de una vez sobre la
misma carpeta, duplicaba también las copias (copias de copias) y la clase crecía mucho más de 3x.

Esta versión:
  - primero BORRA las copias *_dup* que existan en train (deja el dataset como lo exportó Roboflow)
  - luego duplica solo los originales
  - solo toca train (validación y prueba quedan intactas)
  - imprime el antes y el después para que el conteo quede registrado

Uso en Colab:
    from sobremuestreo_idempotente import sobremuestreo_clase
    sobremuestreo_clase(dataset.location, "Potamotrygon_magdalenae", n_copias=3)
"""

import shutil
from pathlib import Path

import yaml

EXT_IMG = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _clases(label_path):
    return {int(float(l.split()[0])) for l in Path(label_path).read_text().splitlines() if l.strip()}


def sobremuestreo_clase(dataset_dir, nombre_clase, n_copias=3):
    dataset_dir = Path(dataset_dir)
    img_dir, lab_dir = dataset_dir / "train" / "images", dataset_dir / "train" / "labels"

    names = yaml.safe_load(open(dataset_dir / "data.yaml", encoding="utf-8"))["names"]
    nombres = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    id_clase = nombres.index(nombre_clase)

    # 1) limpiar copias de ejecuciones anteriores
    previas = [p for p in img_dir.iterdir() if "_dup" in p.stem]
    for p in previas:
        p.unlink()
        lab = lab_dir / f"{p.stem}.txt"
        if lab.exists():
            lab.unlink()
    for lab in lab_dir.glob("*_dup*.txt"):  # etiquetas huérfanas
        lab.unlink()

    antes = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in EXT_IMG)

    # 2) duplicar solo los originales que contienen la clase
    originales = 0
    for lab in sorted(lab_dir.glob("*.txt")):
        if id_clase not in _clases(lab):
            continue
        img = next((img_dir / f"{lab.stem}{e}" for e in EXT_IMG if (img_dir / f"{lab.stem}{e}").exists()), None)
        if img is None:
            continue
        originales += 1
        for i in range(1, n_copias + 1):
            shutil.copy2(img, img_dir / f"{lab.stem}_dup{i}{img.suffix}")
            shutil.copy2(lab, lab_dir / f"{lab.stem}_dup{i}.txt")

    despues = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in EXT_IMG)
    print(f"Clase {nombre_clase} (id {id_clase})")
    print(f"  copias previas eliminadas : {len(previas)}")
    print(f"  imágenes en train ANTES   : {antes}")
    print(f"  originales con la clase   : {originales}  (x{n_copias} copias)")
    print(f"  imágenes en train DESPUÉS : {despues}")
    return {"clase": nombre_clase, "id_clase": id_clase, "copias_previas_eliminadas": len(previas),
            "train_antes": antes, "originales_con_la_clase": originales, "n_copias": n_copias,
            "train_despues": despues}
