"""Ingesta de observaciones validadas por expertos para realimentar el dataset.

Estructura (compatible con YOLO; lista para revisar en Roboflow y reentrenar):
    <FEEDBACK_DIR>/images/<sha256>.jpg       imagen (con la orientación EXIF ya aplicada)
    <FEEDBACK_DIR>/labels/<sha256>.txt       etiqueta YOLO: clase confirmada por el experto + caja del modelo
    <FEEDBACK_DIR>/metadata.jsonl            una línea JSON por observación

IMPORTANTE: la CLASE la pone el experto, pero la CAJA sale del propio modelo (su detección de mayor
confianza). Por eso cada registro lleva requiere_revision_caja = true: revisen las cajas antes de reentrenar.
"""
import hashlib
import io
import json
import os
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

_candado = threading.Lock()


class FeedbackDuplicado(Exception):
    pass


def guardar_feedback(base, imagen: Image.Image, datos_originales: bytes, id_observacion: str,
                     especie: str, class_id: int, prediccion: dict, caja, modelo_sha256,
                     validado_por: str = "") -> dict:
    base = Path(base)
    sha = hashlib.sha256(datos_originales).hexdigest()
    dir_img, dir_lab = base / "images", base / "labels"
    dir_img.mkdir(parents=True, exist_ok=True)
    dir_lab.mkdir(parents=True, exist_ok=True)
    destino = dir_img / f"{sha}.jpg"

    with _candado:
        if destino.exists():
            raise FeedbackDuplicado(sha)
        if caja:
            x, y, w, h = caja["xywhn"]
            (dir_lab / f"{sha}.txt").write_text(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")
        tmp = destino.with_suffix(".tmp")
        imagen.save(tmp, "JPEG", quality=95)
        os.replace(tmp, destino)  # la imagen es lo último que aparece: si algo falla antes, no queda registrada

        registro = {
            "id_observacion": id_observacion,
            "sha256": sha,
            "especie_confirmada": especie,
            "class_id": class_id,
            "validado_por": (validado_por or "").strip()[:100],
            "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "prediccion_modelo": prediccion,
            "caja_fuente": "modelo" if caja else "ninguna",
            "requiere_revision_caja": bool(caja),
            "requiere_anotacion_manual": not caja,
            "modelo_sha256": modelo_sha256,
        }
        with open(base / "metadata.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    return registro


def exportar_zip(base):
    """Empaqueta todo el feedback (imágenes, etiquetas y metadata.jsonl) en un zip; None si aún no hay nada.

    Sirve para respaldar los datos cuando el disco del servidor no es persistente (p. ej. Hugging Face Spaces gratis).
    """
    base = Path(base)
    if not (base / "metadata.jsonl").exists():
        return None
    buf = io.BytesIO()
    with _candado:  # evita leer mientras se está escribiendo una observación
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:  # los JPEG ya están comprimidos
            for ruta in sorted(base.rglob("*")):
                if ruta.is_file() and ruta.suffix != ".tmp":
                    z.write(ruta, ruta.relative_to(base).as_posix())
    return buf.getvalue()
