"""Comprueba que las imágenes NUEVAS no estén ya en el dataset (ni como copia exacta ni como variante casi idéntica).

    python scripts/verificar_duplicados.py --candidatas ./fotos_nuevas --dataset /content/IdentiMar-Rayas-2

Compara con un hash perceptual (dHash de 64 bits) contra todas las imágenes de train/valid/test, también con la
candidata volteada horizontalmente (la aumentación incluye flip). Distancia 0 = idéntica; hasta ~10 = misma foto
re-codificada, recortada o con brillo distinto; ~32 = fotos sin relación. Las sospechosas hay que revisarlas a ojo.
"""
import argparse
import hashlib
import sys
from pathlib import Path

from PIL import Image, ImageOps

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def dhash(img: Image.Image, n: int = 8) -> int:
    g = ImageOps.exif_transpose(img).convert("L").resize((n + 1, n), Image.LANCZOS)
    px = list(g.tobytes())
    bits = 0
    for f in range(n):
        for c in range(n):
            bits = (bits << 1) | int(px[f * (n + 1) + c] > px[f * (n + 1) + c + 1])
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def listar(carpeta: Path):
    return sorted(p for p in Path(carpeta).rglob("*") if p.suffix.lower() in EXT)


def huellas(ruta: Path, n: int = 8):
    datos = ruta.read_bytes()
    with Image.open(ruta) as img:
        img.load()
        return hashlib.sha256(datos).hexdigest(), dhash(img, n), dhash(ImageOps.mirror(img), n)


def comparar(candidatas: Path, dataset: Path, umbral: int = 10, n: int = 8):
    base = []
    for p in listar(dataset):
        try:
            sha, h, _ = huellas(p, n)
            base.append((p, sha, h))
        except Exception:
            continue
    informe = []
    for c in listar(candidatas):
        try:
            sha, h, h_volt = huellas(c, n)
        except Exception as e:
            informe.append({"candidata": c.name, "estado": f"ilegible ({e})", "mas_parecida": "", "distancia": ""})
            continue
        exacta = next((p for p, s, _ in base if s == sha), None)
        if exacta:
            informe.append({"candidata": c.name, "estado": "DUPLICADA EXACTA", "mas_parecida": str(exacta), "distancia": 0})
            continue
        mejor = min(((min(hamming(h, hb), hamming(h_volt, hb)), p) for p, _, hb in base), default=(None, None),
                    key=lambda t: 99 if t[0] is None else t[0])
        d, p = mejor
        estado = "SOSPECHOSA: revisar" if d is not None and d <= umbral else "nueva"
        informe.append({"candidata": c.name, "estado": estado, "mas_parecida": str(p) if p else "", "distancia": d if d is not None else ""})
    return informe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidatas", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--umbral", type=int, default=10, help="distancia máxima (de 64) para marcar una imagen como sospechosa")
    a = ap.parse_args()
    informe = comparar(Path(a.candidatas), Path(a.dataset), a.umbral)
    for f in informe:
        print(f"{f['candidata']:45s} {f['estado']:22s} dist={f['distancia']!s:3s} {f['mas_parecida']}")
    malas = [f for f in informe if f["estado"] != "nueva"]
    print(f"\n{len(informe)} candidatas: {len(informe) - len(malas)} nuevas, {len(malas)} para descartar o revisar.")
    sys.exit(1 if any(f["estado"] == "DUPLICADA EXACTA" for f in informe) else 0)


if __name__ == "__main__":
    main()
