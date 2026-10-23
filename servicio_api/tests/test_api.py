"""Pruebas de la API con un detector simulado (no necesitan torch ni el modelo real).

    pip install pytest httpx
    pytest -q
"""
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main
from app.config import Settings

NOMBRES = {0: "Aetobatus_narinari", 1: "Hypanus_guttatus", 2: "Potamotrygon_magdalenae", 3: "Styracura_schmardae"}


# ---------------------------------------------------------------- detector simulado
class _T:
    def __init__(self, v):
        self.v = v

    def tolist(self):
        return list(self.v)


class _Boxes:
    def __init__(self, conf, cls):
        self.conf, self.cls = _T(conf), _T(cls)
        self.xywhn = _T([[0.5, 0.5, 0.4, 0.3]] * len(conf))
        self.xyxy = _T([[8, 6, 40, 30]] * len(conf))
        self._n = len(conf)

    def __len__(self):
        return self._n


class _Resultado:
    names = NOMBRES

    def __init__(self, conf, cls):
        self.boxes = _Boxes(conf, cls)


class DetectorFalso:
    listo = True
    names = NOMBRES
    sha256 = "a" * 64
    model_path = "falso.pt"

    def __init__(self):
        self.respuesta = _Resultado([0.92], [1])
        self.ultima_imagen = None
        self.falla = False

    def predecir(self, imagen):
        if self.falla:
            raise RuntimeError("boom")
        self.ultima_imagen = imagen
        return self.respuesta


# ---------------------------------------------------------------- utilidades
def imagen_bytes(formato="JPEG", tam=(64, 48), color=(10, 120, 200), exif_orientacion=None):
    buf = io.BytesIO()
    img = Image.new("RGB", tam, color)
    kwargs = {}
    if exif_orientacion:
        exif = Image.Exif()
        exif[0x0112] = exif_orientacion
        kwargs["exif"] = exif
    img.save(buf, formato, **kwargs)
    return buf.getvalue()


def cliente(tmp_path, detector=None, **cfg):
    det = detector or DetectorFalso()
    settings = Settings(feedback_dir=tmp_path / "fb", **cfg)
    app = main.crear_app(detector=det, settings=settings)
    return TestClient(app), det


def post_predict(c, datos=None, id_obs="obs-001", headers=None, nombre="foto.jpg", tipo="image/jpeg"):
    return c.post("/predict", data={"id_observacion": id_obs}, headers=headers or {},
                  files={"imagen": (nombre, datos if datos is not None else imagen_bytes(), tipo)})


# ---------------------------------------------------------------- contrato de /predict
def test_predict_identificado_cumple_el_contrato(tmp_path):
    with cliente(tmp_path)[0] as c:
        r = post_predict(c)
    assert r.status_code == 200
    j = r.json()
    assert {k: j[k] for k in ("id_observacion", "especie_propuesta", "confianza", "umbral_minimo", "estado")} == {
        "id_observacion": "obs-001", "especie_propuesta": "Hypanus_guttatus",
        "confianza": 0.92, "umbral_minimo": 0.7, "estado": "identificado"}
    assert j["version_modelo"] == "yolov8n-identimar-1.0.0+" + "a" * 12 and j["tiempo_inferencia_ms"] >= 0
    assert float(r.headers["X-Tiempo-Inferencia-ms"]) >= 0 and r.headers["X-Modelo-Version"] == j["version_modelo"]


def test_predict_baja_confianza_es_no_concluyente(tmp_path):
    c, det = cliente(tmp_path)
    det.respuesta = _Resultado([0.49], [2])
    with c:
        j = post_predict(c).json()
    assert j["estado"] == "no_concluyente" and j["especie_propuesta"] == "Potamotrygon_magdalenae"


def test_predict_sin_detecciones(tmp_path):
    c, det = cliente(tmp_path)
    det.respuesta = _Resultado([], [])
    with c:
        j = post_predict(c).json()
    assert j["especie_propuesta"] is None and j["confianza"] == 0.0 and j["estado"] == "no_concluyente"


def test_umbral_configurable(tmp_path):
    c, det = cliente(tmp_path, umbral=0.95)
    with c:
        assert post_predict(c).json()["estado"] == "no_concluyente"  # 0.92 < 0.95


def test_la_orientacion_exif_se_aplica(tmp_path):
    c, det = cliente(tmp_path)
    with c:
        r = post_predict(c, imagen_bytes(tam=(64, 48), exif_orientacion=6))
    assert r.status_code == 200 and det.ultima_imagen.size == (48, 64)  # rotada 90 grados


def test_acepta_png_y_webp(tmp_path):
    with cliente(tmp_path)[0] as c:
        assert post_predict(c, imagen_bytes("PNG"), nombre="f.png", tipo="image/png").status_code == 200
        assert post_predict(c, imagen_bytes("WEBP"), nombre="f.webp", tipo="image/webp").status_code == 200


# ---------------------------------------------------------------- validaciones
@pytest.mark.parametrize("id_malo", ["", "../etc/passwd", "a b", "x" * 65, "obs;rm", "-abc"])
def test_id_observacion_invalido(tmp_path, id_malo):
    with cliente(tmp_path)[0] as c:
        r = post_predict(c, id_obs=id_malo)
    assert r.status_code in (400, 422)


def test_archivo_que_no_es_imagen(tmp_path):
    with cliente(tmp_path)[0] as c:
        assert post_predict(c, b"esto no es una imagen").status_code == 400


def test_archivo_vacio(tmp_path):
    with cliente(tmp_path)[0] as c:
        assert post_predict(c, b"").status_code == 400


def test_formato_no_permitido(tmp_path):
    with cliente(tmp_path)[0] as c:
        r = post_predict(c, imagen_bytes("GIF"), nombre="f.gif", tipo="image/gif")
    assert r.status_code == 400 and "no soportado" in r.json()["detail"]


def test_imagen_truncada(tmp_path):
    datos = imagen_bytes(tam=(400, 300))
    with cliente(tmp_path)[0] as c:
        assert post_predict(c, datos[: len(datos) // 2]).status_code == 400


def test_imagen_muy_pequena(tmp_path):
    with cliente(tmp_path)[0] as c:
        assert post_predict(c, imagen_bytes(tam=(10, 10))).status_code == 400


def test_archivo_demasiado_pesado(tmp_path):
    with cliente(tmp_path, max_upload_mb=0.0005)[0] as c:  # ~500 bytes
        assert post_predict(c, imagen_bytes(tam=(300, 300))).status_code == 413


def test_imagen_con_demasiados_pixeles(tmp_path):
    with cliente(tmp_path, max_pixels=1000)[0] as c:
        assert post_predict(c, imagen_bytes(tam=(64, 48))).status_code == 413


def test_falta_la_imagen(tmp_path):
    with cliente(tmp_path)[0] as c:
        assert c.post("/predict", data={"id_observacion": "x"}).status_code == 422


def test_error_de_inferencia_devuelve_500_sin_detalles_internos(tmp_path):
    c, det = cliente(tmp_path)
    det.falla = True
    with c:
        r = post_predict(c)
    assert r.status_code == 500 and "boom" not in r.text


def test_modelo_no_cargado_devuelve_503(tmp_path):
    det = DetectorFalso()
    det.listo = False
    with cliente(tmp_path, detector=det)[0] as c:
        assert post_predict(c).status_code == 503
        assert c.get("/health").status_code == 503


# ---------------------------------------------------------------- autenticación y salud
def test_health(tmp_path):
    with cliente(tmp_path)[0] as c:
        j = c.get("/health").json()
    assert j["estado"] == "ok" and j["modelo"]["clases"] == list(NOMBRES.values())


def test_api_key(tmp_path):
    c, _ = cliente(tmp_path, api_key="secreto")
    with c:
        assert post_predict(c).status_code == 401
        assert post_predict(c, headers={"X-API-Key": "otra"}).status_code == 401
        assert post_predict(c, headers={"X-API-Key": "secreto"}).status_code == 200
        assert c.get("/health").status_code == 200  # la salud no exige clave (healthcheck de Docker)


def test_api_key_se_exige_antes_que_el_modelo(tmp_path):
    det = DetectorFalso()
    det.listo = False
    c, _ = cliente(tmp_path, detector=det, api_key="secreto")
    with c:
        assert post_predict(c).status_code == 401


def test_openapi_expone_el_contrato(tmp_path):
    with cliente(tmp_path)[0] as c:
        esquema = c.get("/openapi.json").json()
    props = esquema["components"]["schemas"]["RespuestaIdentificacion"]["properties"]
    assert set(props) == {"id_observacion", "especie_propuesta", "confianza", "umbral_minimo", "estado",
                          "version_modelo", "tiempo_inferencia_ms"}


# ---------------------------------------------------------------- feedback
def post_feedback(c, datos=None, id_obs="obs-100", especie="Hypanus_guttatus", experto="Dra. Ejemplo"):
    return c.post("/feedback", data={"id_observacion": id_obs, "especie_confirmada": especie, "validado_por": experto},
                  files={"imagen": ("f.jpg", datos if datos is not None else imagen_bytes(), "image/jpeg")})


def test_feedback_guarda_imagen_etiqueta_y_metadata(tmp_path):
    c, det = cliente(tmp_path)
    det.respuesta = _Resultado([0.55], [2])  # el modelo dijo Magdalena; el experto confirma Hypanus
    with c:
        r = post_feedback(c)
    assert r.status_code == 201
    j = r.json()
    base = tmp_path / "fb"
    assert (base / "images" / f"{j['sha256']}.jpg").exists()
    assert (base / "labels" / f"{j['sha256']}.txt").read_text().split()[0] == "1"  # clase del EXPERTO (Hypanus = 1)
    reg = json.loads((base / "metadata.jsonl").read_text().splitlines()[0])
    assert reg["especie_confirmada"] == "Hypanus_guttatus" and reg["prediccion_modelo"]["especie_propuesta"] == \
        "Potamotrygon_magdalenae" and reg["requiere_revision_caja"] is True and reg["caja_fuente"] == "modelo"


def test_feedback_sin_deteccion_queda_para_anotacion_manual(tmp_path):
    c, det = cliente(tmp_path)
    det.respuesta = _Resultado([], [])
    with c:
        j = post_feedback(c).json()
    base = tmp_path / "fb"
    assert j["caja_del_modelo"] is False and not (base / "labels" / f"{j['sha256']}.txt").exists()
    reg = json.loads((base / "metadata.jsonl").read_text().splitlines()[0])
    assert reg["requiere_anotacion_manual"] is True


def test_feedback_rechaza_duplicados(tmp_path):
    with cliente(tmp_path)[0] as c:
        datos = imagen_bytes(color=(1, 2, 3))
        assert post_feedback(c, datos).status_code == 201
        assert post_feedback(c, datos, id_obs="otro").status_code == 409
    assert len((tmp_path / "fb" / "metadata.jsonl").read_text().splitlines()) == 1


def test_feedback_acepta_nombre_con_espacios_y_mayusculas(tmp_path):
    with cliente(tmp_path)[0] as c:
        assert post_feedback(c, especie="hypanus guttatus").json()["especie_confirmada"] == "Hypanus_guttatus"


def test_feedback_especie_desconocida(tmp_path):
    with cliente(tmp_path)[0] as c:
        r = post_feedback(c, especie="Tiburon_blanco")
    assert r.status_code == 400 and "Hypanus_guttatus" in r.json()["detail"]
    assert not (tmp_path / "fb").exists()


def test_feedback_con_id_peligroso(tmp_path):
    with cliente(tmp_path)[0] as c:
        assert post_feedback(c, id_obs="../../x").status_code in (400, 422)


def test_feedback_si_falla_la_inferencia_igual_se_guarda(tmp_path):
    c, det = cliente(tmp_path)
    det.falla = True
    with c:
        r = post_feedback(c)
    assert r.status_code == 201 and r.json()["caja_del_modelo"] is False


def test_feedback_requiere_api_key(tmp_path):
    c, _ = cliente(tmp_path, api_key="secreto")
    with c:
        assert post_feedback(c).status_code == 401


# ---------------------------------------------------------------- /predict/url
def test_url_deshabilitada_por_defecto(tmp_path):
    with cliente(tmp_path)[0] as c:
        r = c.post("/predict/url", json={"id_observacion": "x", "url_imagen": "https://almacen.test/a.jpg"})
    assert r.status_code == 501


def test_url_con_host_permitido(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "descargar_imagen", lambda url, hosts, mx, t: imagen_bytes())
    with cliente(tmp_path, hosts_permitidos=["almacen.test"])[0] as c:
        r = c.post("/predict/url", json={"id_observacion": "x", "url_imagen": "https://almacen.test/a.jpg"})
    assert r.status_code == 200 and r.json()["estado"] == "identificado"


@pytest.mark.parametrize("url,codigo", [("http://almacen.test/a.jpg", 400), ("https://malo.test/a.jpg", 403),
                                         ("https://almacen.test@malo.test/a.jpg", 403),
                                         ("https://127.0.0.1/a.jpg", 403)])
def test_url_bloquea_destinos_no_permitidos(tmp_path, url, codigo):
    with cliente(tmp_path, hosts_permitidos=["almacen.test"])[0] as c:
        r = c.post("/predict/url", json={"id_observacion": "x", "url_imagen": url})
    assert r.status_code == codigo


# ---------------------------------------------------------------- /feedback/export
def test_export_sin_datos_devuelve_404(tmp_path):
    with cliente(tmp_path)[0] as c:
        assert c.get("/feedback/export").status_code == 404


def test_export_devuelve_un_zip_con_todo(tmp_path):
    import zipfile
    with cliente(tmp_path)[0] as c:
        sha = post_feedback(c).json()["sha256"]
        r = c.get("/feedback/export")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    nombres = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert {"metadata.jsonl", f"images/{sha}.jpg", f"labels/{sha}.txt"} <= set(nombres)


def test_export_requiere_api_key(tmp_path):
    c, _ = cliente(tmp_path, api_key="secreto")
    with c:
        assert c.get("/feedback/export").status_code == 401
        assert c.get("/feedback/export", headers={"X-API-Key": "secreto"}).status_code == 404  # autorizado, sin datos


# ---------------------------------------------------------------- Hugging Face Hub (opcional)
import app.hf as hf


def test_feedback_se_sincroniza_con_el_hub_si_esta_configurado(tmp_path, monkeypatch):
    subidas = []
    monkeypatch.setattr(hf, "subir_feedback", lambda repo, token, archivos, msg: subidas.append((repo, token, archivos, msg)))
    c, _ = cliente(tmp_path, hf_token="tk", hf_feedback_repo="usuario/feedback")
    with c:
        j = post_feedback(c).json()
    assert len(subidas) == 1
    repo, token, archivos, msg = subidas[0]
    sha = j["sha256"]
    assert repo == "usuario/feedback" and token == "tk" and "obs-100" in msg
    assert set(archivos) == {f"images/{sha}.jpg", f"labels/{sha}.txt", f"meta/{sha}.json"}
    assert json.loads(archivos[f"meta/{sha}.json"])["especie_confirmada"] == "Hypanus_guttatus"


def test_feedback_no_sincroniza_sin_configuracion(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "subir_feedback", lambda *a: pytest.fail("no debía subir nada"))
    with cliente(tmp_path)[0] as c:
        assert post_feedback(c).status_code == 201


def test_si_falla_la_subida_el_feedback_igual_queda_guardado(tmp_path, monkeypatch):
    def falla(*a):
        raise RuntimeError("sin red")
    monkeypatch.setattr(hf, "subir_feedback", falla)
    c, _ = cliente(tmp_path, hf_token="tk", hf_feedback_repo="usuario/feedback")
    with c:
        r = post_feedback(c)
    assert r.status_code == 201 and (tmp_path / "fb" / "metadata.jsonl").exists()


def test_health_informa_si_el_feedback_es_remoto(tmp_path):
    c, _ = cliente(tmp_path, hf_token="tk", hf_feedback_repo="usuario/feedback")
    with c:
        assert c.get("/health").json()["feedback_remoto"] is True


def test_modelo_se_descarga_del_hub_solo_si_falta(tmp_path, monkeypatch):
    llamadas = []

    def falso(repo, archivo, destino, token, rev):
        llamadas.append((repo, archivo, rev))
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(b"pesos")
        return destino

    monkeypatch.setattr(hf, "descargar_modelo", falso)
    cfg = Settings(model_path=tmp_path / "m" / "best.pt", hf_model_repo="usuario/modelo", hf_token="tk",
                   hf_model_revision="abc123")
    hf.asegurar_modelo(cfg)
    hf.asegurar_modelo(cfg)  # la segunda vez ya existe: no vuelve a descargar
    assert llamadas == [("usuario/modelo", "best.pt", "abc123")] and cfg.model_path.read_bytes() == b"pesos"


def test_sin_repositorio_configurado_no_se_descarga_nada(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "descargar_modelo", lambda *a: pytest.fail("no debía descargar"))
    hf.asegurar_modelo(Settings(model_path=tmp_path / "no_existe.pt"))


def test_subir_feedback_arma_un_solo_commit(monkeypatch):
    """Verifica la forma de la llamada con un huggingface_hub simulado (las firmas reales se comprobaron aparte)."""
    import sys, types
    registro = {}

    class FakeApi:
        def __init__(self, token=None):
            registro["token"] = token

        def create_repo(self, **kw):
            registro["repo"] = kw

        def create_commit(self, **kw):
            registro["commit"] = kw

    fake = types.SimpleNamespace(HfApi=FakeApi, CommitOperationAdd=lambda path_in_repo, path_or_fileobj: (path_in_repo, path_or_fileobj))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setattr(hf, "_repos_creados", set())
    hf.subir_feedback("u/r", "tk", {"images/a.jpg": b"x", "meta/a.json": b"{}"}, "msg")
    assert registro["repo"]["private"] is True and registro["repo"]["repo_type"] == "dataset" and registro["token"] == "tk"
    assert [op[0] for op in registro["commit"]["operations"]] == ["images/a.jpg", "meta/a.json"]


# ---------------------------------------------------------------- v1.1: versión, catálogo, visual, demo
def test_especies_devuelve_el_catalogo(tmp_path):
    with cliente(tmp_path)[0] as c:
        j = c.get("/especies").json()
    assert j["especies"][1] == {"id": 1, "clase": "Hypanus_guttatus", "nombre_cientifico": "Hypanus guttatus",
                                "nombre_comun": "Raya látigo"}
    assert len(j["especies"]) == 4 and j["version_modelo"].endswith("a" * 12)


def test_predict_visual_devuelve_jpeg_y_el_resultado_en_el_encabezado(tmp_path):
    with cliente(tmp_path)[0] as c:
        r = c.post("/predict/visual", data={"id_observacion": "v1"},
                   files={"imagen": ("f.jpg", imagen_bytes(tam=(200, 150)), "image/jpeg")})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    img = Image.open(io.BytesIO(r.content))
    assert img.size == (200, 150) and img.format == "JPEG"
    assert json.loads(r.headers["X-Resultado"])["especie_propuesta"] == "Hypanus_guttatus"
    # las cajas se dibujaron: el color del borde de la caja (8,6)-(40,30) ya no es el del fondo
    assert img.convert("RGB").getpixel((20, 6)) != (10, 120, 200)


def test_predict_visual_sin_detecciones_devuelve_la_imagen_sin_cajas(tmp_path):
    c, det = cliente(tmp_path)
    det.respuesta = _Resultado([], [])
    with c:
        r = c.post("/predict/visual", data={"id_observacion": "v2"},
                   files={"imagen": ("f.jpg", imagen_bytes(), "image/jpeg")})
    assert r.status_code == 200 and json.loads(r.headers["X-Resultado"])["estado"] == "no_concluyente"


def test_predict_visual_y_especies_exigen_api_key(tmp_path):
    c, _ = cliente(tmp_path, api_key="secreto")
    with c:
        assert c.get("/especies").status_code == 401
        assert c.post("/predict/visual", data={"id_observacion": "v"},
                      files={"imagen": ("f.jpg", imagen_bytes(), "image/jpeg")}).status_code == 401


def test_pagina_demo_y_raiz(tmp_path):
    with cliente(tmp_path, api_key="secreto")[0] as c:
        d = c.get("/demo")
        assert d.status_code == 200 and "predict/visual" in d.text and d.headers["content-type"].startswith("text/html")
        assert c.get("/").json()["demostracion"] == "/demo"
