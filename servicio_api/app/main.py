"""API REST del módulo de visión artificial de IdentiMar.

Flujo: Backend (Equipo 3) -> POST /predict -> JSON del contrato -> Backend -> App móvil (Equipo 1).
El Equipo 1 nunca habla con este servicio: solo ve el resultado normalizado que le reenvía el backend.
"""
import io
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi.responses import HTMLResponse
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field

from app import __version__
from app.almacenamiento import descargar_imagen
from app.config import Settings
from app.contrato_salida import generar_respuesta
from app.demo import PAGINA_DEMO
from app.detector import Detector, mejor_caja
from app.especies import catalogo
from app.visual import anotar
from app.hf import asegurar_modelo, sincronizar_feedback
from app.feedback import FeedbackDuplicado, exportar_zip, guardar_feedback
from app.imagenes import ImagenInvalida, ImagenMuyGrande, decodificar_imagen
from app.validacion import validar_id

logger = logging.getLogger("identimar")

DESCRIPCION = """
Servicio de identificación de rayas del Urabá antioqueño (YOLOv8).

**Contrato de salida** de `/predict` (JSON): `id_observacion`, `especie_propuesta`, `confianza`,
`umbral_minimo` y `estado` (`identificado` | `no_concluyente`). Cuando la confianza es menor al umbral,
el estado es `no_concluyente` y el backend deriva la observación a revisión por un experto.
Si el modelo no detecta ninguna raya, `especie_propuesta` es `null`, `confianza` es 0 y el estado es `no_concluyente`.

El tiempo de inferencia de cada petición viaja en el encabezado `X-Tiempo-Inferencia-ms`
(no en el cuerpo, para no alterar el contrato).
"""


class RespuestaIdentificacion(BaseModel):
    id_observacion: str
    especie_propuesta: Optional[str] = Field(None, description="Nombre de la clase; null si no hubo detección.")
    confianza: float = Field(..., description="Confianza de la detección propuesta (0 a 1).")
    umbral_minimo: float = Field(..., description="Umbral del contrato; por debajo de él el estado es no_concluyente.")
    estado: Literal["identificado", "no_concluyente"]
    version_modelo: str = Field(..., description="Versión del modelo que produjo la respuesta (nombre + hash corto de los pesos).")
    tiempo_inferencia_ms: float = Field(..., description="Tiempo de inferencia del modelo en el servidor, en milisegundos.")

    model_config = {"json_schema_extra": {"example": {
        "id_observacion": "obs-0001", "especie_propuesta": "Hypanus_guttatus",
        "confianza": 0.9132, "umbral_minimo": 0.7, "estado": "identificado",
        "version_modelo": "yolov8n-identimar-1.0.0+3fa9c01b22de", "tiempo_inferencia_ms": 142.3}}}


class PeticionUrl(BaseModel):
    id_observacion: str
    url_imagen: str = Field(..., description="URL https de la imagen en un almacenamiento permitido.")


class RespuestaFeedback(BaseModel):
    estado: Literal["almacenado"]
    id_observacion: str
    sha256: str
    especie_confirmada: str
    caja_del_modelo: bool = Field(..., description="True si la etiqueta incluye una caja propuesta por el modelo.")
    requiere_revision_caja: bool


def crear_app(detector=None, settings: Optional[Settings] = None) -> FastAPI:
    """Fábrica de la app. En producción se llama sin argumentos; las pruebas inyectan un detector simulado."""
    cfg = settings or Settings.desde_entorno()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if detector is not None:
            app.state.detector = detector
        else:
            det = Detector(cfg.model_path, cfg.conf_predict, cfg.imgsz)
            try:
                asegurar_modelo(cfg)
                det.cargar()
                logger.info("Modelo cargado: %s (sha256 %s)", cfg.model_path, det.sha256)
            except Exception:
                logger.exception("No se pudo cargar el modelo; /predict responderá 503.")
            app.state.detector = det
        if not cfg.api_key:
            logger.warning("API_KEY vacía: el servicio no exige autenticación (solo para desarrollo).")
        yield

    app = FastAPI(title="IdentiMar — Módulo de visión artificial", version=__version__,
                  description=DESCRIPCION, lifespan=lifespan)

    # ------------------------------------------------------------------ dependencias
    def verificar_api_key(x_api_key: Optional[str] = Header(default=None)):
        if cfg.api_key and not (x_api_key and secrets.compare_digest(x_api_key, cfg.api_key)):
            raise HTTPException(status_code=401, detail="API key inválida o ausente (encabezado X-API-Key).")

    def obtener_detector():
        det = app.state.detector
        if not det.listo:
            raise HTTPException(status_code=503, detail="El modelo no está cargado.")
        return det

    # ------------------------------------------------------------------ utilidades
    def decodificar(datos: bytes):
        if not datos:
            raise HTTPException(status_code=400, detail="El archivo de imagen está vacío.")
        if len(datos) > cfg.max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"La imagen supera el máximo de {cfg.max_upload_mb:g} MB.")
        try:
            return decodificar_imagen(datos, cfg.max_pixels)
        except ImagenMuyGrande as e:
            raise HTTPException(status_code=413, detail=str(e))
        except ImagenInvalida as e:
            raise HTTPException(status_code=400, detail=str(e))

    def inferir(det, id_obs: str, datos: bytes):
        imagen = decodificar(datos)
        t0 = time.perf_counter()
        try:
            resultado = det.predecir(imagen)
        except Exception:
            logger.exception("Falló la inferencia (id=%s)", id_obs)
            raise HTTPException(status_code=500, detail="Error interno durante la inferencia.")
        ms = (time.perf_counter() - t0) * 1000
        respuesta = generar_respuesta(resultado, id_obs, cfg.umbral)
        respuesta["version_modelo"] = f"{cfg.model_version}+{(det.sha256 or 'sin-hash')[:12]}"
        respuesta["tiempo_inferencia_ms"] = round(ms, 1)
        logger.info("predict id=%s estado=%s especie=%s confianza=%.3f ms=%.1f", id_obs,
                    respuesta["estado"], respuesta["especie_propuesta"], respuesta["confianza"], ms)
        return imagen, resultado, respuesta, ms

    def identificar(response: Response, det, id_obs: str, datos: bytes) -> dict:
        _, _, respuesta, ms = inferir(det, id_obs, datos)
        response.headers["X-Tiempo-Inferencia-ms"] = f"{ms:.1f}"
        response.headers["X-Modelo-Version"] = respuesta["version_modelo"]
        return respuesta

    # ------------------------------------------------------------------ endpoints
    @app.get("/health", summary="Estado del servicio")
    def health():
        det = app.state.detector
        cuerpo = {"estado": "ok" if det.listo else "modelo_no_cargado", "version_api": __version__,
                  "umbral_minimo": cfg.umbral, "autenticacion": bool(cfg.api_key),
                  "url_habilitada": bool(cfg.hosts_permitidos), "feedback_remoto": bool(cfg.hf_feedback_repo and cfg.hf_token),
                  "modelo": {"archivo": str(det.model_path) if hasattr(det, "model_path") else None,
                             "sha256": det.sha256, "clases": list(det.names.values())}}
        if not det.listo:
            raise HTTPException(status_code=503, detail=cuerpo)
        return cuerpo

    @app.post("/predict", response_model=RespuestaIdentificacion, dependencies=[Depends(verificar_api_key)],
              summary="Identificar la raya de una imagen enviada como archivo",
              responses={400: {"description": "Imagen inválida o id_observacion inválido"},
                         401: {"description": "API key inválida"}, 413: {"description": "Imagen demasiado grande"},
                         503: {"description": "Modelo no cargado"}})
    def predict(response: Response, id_observacion: str = Form(..., description="Identificador de la observación."),
                imagen: UploadFile = File(..., description="Foto en JPEG, PNG o WEBP."),
                det=Depends(obtener_detector)):
        id_obs = validar_id(id_observacion)
        datos = imagen.file.read(cfg.max_upload_bytes + 1)
        return identificar(response, det, id_obs, datos)

    @app.post("/predict/url", response_model=RespuestaIdentificacion, dependencies=[Depends(verificar_api_key)],
              summary="Identificar la raya de una imagen referenciada por URL (opcional)",
              description="Deshabilitado salvo que se configure `ALLOWED_IMAGE_HOSTS` (hosts de almacenamiento permitidos).",
              responses={403: {"description": "Host no permitido"}, 501: {"description": "Función deshabilitada"},
                         502: {"description": "El almacenamiento no respondió bien"}})
    def predict_url(response: Response, peticion: PeticionUrl, det=Depends(obtener_detector)):
        if not cfg.hosts_permitidos:
            raise HTTPException(status_code=501, detail="Referencia por URL deshabilitada: configure ALLOWED_IMAGE_HOSTS.")
        id_obs = validar_id(peticion.id_observacion)
        try:
            datos = descargar_imagen(peticion.url_imagen, cfg.hosts_permitidos, cfg.max_upload_bytes, cfg.url_timeout_s)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e))
        except OverflowError as e:
            raise HTTPException(status_code=413, detail=str(e))
        except Exception:
            logger.exception("No se pudo descargar la imagen (id=%s)", id_obs)
            raise HTTPException(status_code=502, detail="No se pudo obtener la imagen desde el almacenamiento.")
        return identificar(response, det, id_obs, datos)

    @app.post("/predict/visual", dependencies=[Depends(verificar_api_key)],
              summary="Imagen con las detecciones dibujadas (demostración)",
              description="Devuelve un JPEG con las cajas, la especie y la confianza. El resultado del contrato viaja en el "
                          "encabezado `X-Resultado` (JSON).", responses={200: {"content": {"image/jpeg": {}}}})
    def predict_visual(id_observacion: str = Form(...), imagen: UploadFile = File(...), det=Depends(obtener_detector)):
        id_obs = validar_id(id_observacion)
        datos = imagen.file.read(cfg.max_upload_bytes + 1)
        img, resultado, respuesta, ms = inferir(det, id_obs, datos)
        buf = io.BytesIO()
        anotar(img, resultado, det.names).save(buf, "JPEG", quality=90)
        return Response(content=buf.getvalue(), media_type="image/jpeg",
                        headers={"X-Resultado": json.dumps(respuesta, ensure_ascii=True, separators=(",", ":")),
                                 "X-Tiempo-Inferencia-ms": f"{ms:.1f}"})

    @app.get("/especies", dependencies=[Depends(verificar_api_key)],
             summary="Catálogo de especies que reconoce el modelo",
             description="Úselo para armonizar nombres científicos y códigos con el backend y la app.")
    def especies(det=Depends(obtener_detector)):
        return {"version_modelo": f"{cfg.model_version}+{(det.sha256 or 'sin-hash')[:12]}", "especies": catalogo(det.names)}

    @app.get("/demo", include_in_schema=False, response_class=HTMLResponse)
    def demo():
        return PAGINA_DEMO

    @app.get("/", include_in_schema=False)
    def raiz():
        return {"servicio": "IdentiMar — visión artificial", "version_api": __version__,
                "documentacion": "/docs", "estado": "/health", "demostracion": "/demo"}

    @app.post("/feedback", status_code=201, response_model=RespuestaFeedback,
              dependencies=[Depends(verificar_api_key)],
              summary="Ingresar una observación validada por un experto (realimentación del dataset)",
              description="Guarda la imagen y una etiqueta YOLO con la especie confirmada. La caja sale del modelo y "
                          "debe revisarse antes de reentrenar (`requiere_revision_caja`).",
              responses={400: {"description": "Especie o imagen inválida"}, 409: {"description": "Imagen ya ingresada"}})
    def feedback(id_observacion: str = Form(...),
                 especie_confirmada: str = Form(..., description="Clase confirmada por el experto, p. ej. Hypanus_guttatus."),
                 validado_por: str = Form("", description="Nombre o identificador del experto (opcional)."),
                 imagen: UploadFile = File(...), det=Depends(obtener_detector)):
        id_obs = validar_id(id_observacion)
        buscada = especie_confirmada.strip().replace(" ", "_").lower()
        coincide = [(i, n) for i, n in det.names.items() if n.lower() == buscada]
        if not coincide:
            raise HTTPException(status_code=400, detail=f"Especie desconocida. Opciones: {sorted(det.names.values())}")
        class_id, especie = coincide[0]

        datos = imagen.file.read(cfg.max_upload_bytes + 1)
        img = decodificar(datos)
        try:
            resultado = det.predecir(img)
            caja, prediccion = mejor_caja(resultado), generar_respuesta(resultado, id_obs, cfg.umbral)
        except Exception:  # sin caja del modelo, la observación queda para anotación manual
            logger.exception("Inferencia fallida al ingresar feedback (id=%s)", id_obs)
            caja, prediccion = None, None
        try:
            reg = guardar_feedback(cfg.feedback_dir, img, datos, id_obs, especie, int(class_id), prediccion,
                                   caja, det.sha256, validado_por)
        except FeedbackDuplicado:
            raise HTTPException(status_code=409, detail="Esta imagen ya fue ingresada.")
        if cfg.hf_feedback_repo and cfg.hf_token:
            sincronizar_feedback(cfg, reg)  # antes de responder; un fallo se registra pero no cambia la respuesta
        logger.info("feedback id=%s especie=%s caja=%s", id_obs, especie, bool(caja))
        return {"estado": "almacenado", "id_observacion": id_obs, "sha256": reg["sha256"],
                "especie_confirmada": especie, "caja_del_modelo": bool(caja),
                "requiere_revision_caja": reg["requiere_revision_caja"]}

    @app.get("/feedback/export", dependencies=[Depends(verificar_api_key)],
             summary="Descargar todo el feedback almacenado (zip)",
             description="Devuelve images/, labels/ y metadata.jsonl. Úselo para respaldar los datos si el disco del "
                         "servidor no es persistente.",
             responses={200: {"content": {"application/zip": {}}}, 404: {"description": "Aún no hay observaciones"}})
    def feedback_export():
        datos = exportar_zip(cfg.feedback_dir)
        if datos is None:
            raise HTTPException(status_code=404, detail="Aún no hay observaciones almacenadas.")
        return Response(content=datos, media_type="application/zip",
                        headers={"Content-Disposition": "attachment; filename=feedback.zip"})

    return app


app = crear_app()
