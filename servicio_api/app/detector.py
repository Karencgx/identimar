"""Envoltura del modelo YOLO: se carga una vez y las inferencias se serializan con un candado."""
import hashlib
import threading
from pathlib import Path

from PIL import Image


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


class Detector:
    def __init__(self, model_path, conf: float = 0.25, imgsz: int = 640):
        self.model_path = Path(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.names = {}
        self.sha256 = None
        self._modelo = None
        self._candado = threading.Lock()  # un modelo YOLO no debe usarse desde varios hilos a la vez

    @property
    def listo(self) -> bool:
        return self._modelo is not None

    def cargar(self):
        from ultralytics import YOLO  # import tardío: no hace falta para las pruebas con detector simulado

        if not self.model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo en {self.model_path}")
        self._modelo = YOLO(str(self.model_path))
        self.names = dict(self._modelo.names)
        self.sha256 = _sha256(self.model_path)
        self.predecir(Image.new("RGB", (self.imgsz, self.imgsz)))  # calentamiento: la 1a inferencia es lenta

    def predecir(self, imagen: Image.Image):
        with self._candado:
            return self._modelo.predict(imagen, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]


def mejor_caja(resultado):
    """Caja (x, y, ancho, alto normalizados) de la detección de mayor confianza, o None."""
    if len(resultado.boxes) == 0:
        return None
    confs = resultado.boxes.conf.tolist()
    i = max(range(len(confs)), key=confs.__getitem__)
    return {"xywhn": tuple(resultado.boxes.xywhn.tolist()[i]), "confianza": float(confs[i])}
