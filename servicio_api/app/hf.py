"""Integración OPCIONAL con Hugging Face Hub. Sin las variables de entorno correspondientes no hace nada.

1) Modelo privado: si el modelo no está en disco, se descarga al arrancar desde un repositorio PRIVADO del Hub
   (HF_MODEL_REPO + HF_TOKEN). Así el archivo best.pt no queda en el repositorio público del Space.
2) Feedback persistente: cada observación validada se sube a un dataset PRIVADO (HF_FEEDBACK_REPO + HF_TOKEN),
   porque el disco de un Space gratuito se borra al reiniciar.
"""
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger("identimar")
_repos_creados = set()


def descargar_modelo(repo_id, filename, destino: Path, token, revision=None) -> Path:
    from huggingface_hub import hf_hub_download  # import tardío: solo se necesita si se usa el Hub

    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    ruta = Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model", revision=revision or None,
                                token=token or None, local_dir=str(destino.parent)))
    if ruta.resolve() != destino.resolve():
        shutil.copy2(ruta, destino)
    return destino


def asegurar_modelo(cfg):
    """Si el modelo no está en disco y hay un repositorio configurado, lo descarga."""
    if cfg.model_path.exists() or not cfg.hf_model_repo:
        return
    logger.info("Descargando el modelo desde el repositorio privado %s ...", cfg.hf_model_repo)
    descargar_modelo(cfg.hf_model_repo, cfg.hf_model_file, cfg.model_path, cfg.hf_token, cfg.hf_model_revision)


def subir_feedback(repo_id, token, archivos: dict, mensaje: str):
    """Sube varios archivos al dataset privado en UN solo commit."""
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    if repo_id not in _repos_creados:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        _repos_creados.add(repo_id)
    operaciones = [CommitOperationAdd(path_in_repo=ruta, path_or_fileobj=contenido) for ruta, contenido in archivos.items()]
    api.create_commit(repo_id=repo_id, repo_type="dataset", operations=operaciones, commit_message=mensaje)


def sincronizar_feedback(cfg, registro: dict):
    """Sube la observación al Hub. Se llama antes de responder (en plataformas serverless el trabajo posterior a la respuesta puede quedar congelado). Un fallo se registra pero no afecta la respuesta."""
    sha = registro["sha256"]
    base = Path(cfg.feedback_dir)
    try:
        archivos = {f"images/{sha}.jpg": (base / "images" / f"{sha}.jpg").read_bytes(),
                    f"meta/{sha}.json": json.dumps(registro, ensure_ascii=False, indent=1).encode("utf-8")}
        etiqueta = base / "labels" / f"{sha}.txt"
        if etiqueta.exists():
            archivos[f"labels/{sha}.txt"] = etiqueta.read_bytes()
        subir_feedback(cfg.hf_feedback_repo, cfg.hf_token, archivos, f"feedback {registro['id_observacion']}")
        logger.info("feedback %s sincronizado con %s", registro["id_observacion"], cfg.hf_feedback_repo)
    except Exception:
        logger.exception("No se pudo sincronizar el feedback %s con el Hub (queda guardado en disco)",
                         registro.get("id_observacion"))
