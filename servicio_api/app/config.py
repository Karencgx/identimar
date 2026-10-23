"""Configuración del servicio: todo se controla con variables de entorno."""
import os
from dataclasses import dataclass, field
from pathlib import Path


def _lista(valor: str):
    return [h.strip().lower() for h in valor.split(",") if h.strip()]


@dataclass
class Settings:
    model_path: Path = Path("/models/best.pt")
    umbral: float = 0.70             # confianza < umbral -> no_concluyente (contrato con el Equipo 3)
    conf_predict: float = 0.25       # piso con el que predice el modelo (ver contrato_salida.py)
    imgsz: int = 640
    max_upload_mb: float = 10.0
    max_pixels: int = 50_000_000     # protege contra imágenes "bomba" que agotan la memoria
    api_key: str = ""                # vacío = sin autenticación (solo para desarrollo)
    feedback_dir: Path = Path("/data/feedback")
    model_version: str = "yolov8n-identimar-1.0.0"   # nombre de la versión; se le agrega el hash corto de los pesos
    hosts_permitidos: list = field(default_factory=list)  # vacío = /predict/url deshabilitado
    url_timeout_s: float = 10.0
    # Hugging Face Hub (opcional)
    hf_token: str = ""
    hf_model_repo: str = ""          # p. ej. usuario/identimar-modelo (repositorio PRIVADO con best.pt)
    hf_model_file: str = "best.pt"
    hf_model_revision: str = ""      # commit exacto del modelo, para congelar la versión
    hf_feedback_repo: str = ""       # p. ej. usuario/identimar-feedback (dataset PRIVADO)

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @classmethod
    def desde_entorno(cls) -> "Settings":
        e = os.environ.get
        return cls(
            model_path=Path(e("MODEL_PATH", "/models/best.pt")),
            umbral=float(e("UMBRAL_CONFIANZA", "0.70")),
            conf_predict=float(e("CONF_PREDICT", "0.25")),
            imgsz=int(e("IMGSZ", "640")),
            max_upload_mb=float(e("MAX_UPLOAD_MB", "10")),
            max_pixels=int(e("MAX_PIXELS", "50000000")),
            api_key=e("API_KEY", ""),
            feedback_dir=Path(e("FEEDBACK_DIR", "/data/feedback")),
            model_version=e("MODEL_VERSION", "yolov8n-identimar-1.0.0"),
            hosts_permitidos=_lista(e("ALLOWED_IMAGE_HOSTS", "")),
            url_timeout_s=float(e("URL_TIMEOUT_S", "10")),
            hf_token=e("HF_TOKEN", ""),
            hf_model_repo=e("HF_MODEL_REPO", ""),
            hf_model_file=e("HF_MODEL_FILE", "best.pt"),
            hf_model_revision=e("HF_MODEL_REVISION", ""),
            hf_feedback_repo=e("HF_FEEDBACK_REPO", ""),
        )
