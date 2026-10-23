"""Pruebas de los scripts de apoyo del equipo (duplicados, evaluación por grupos, ingesta de validadas)."""
import csv
import io
import json
import random
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluar_externas as ee  # noqa: E402
import ingerir_validadas as iv  # noqa: E402
import verificar_duplicados as vd  # noqa: E402
from tests.test_api import NOMBRES, _Resultado  # noqa: E402


def foto_sintetica(semilla, tam=(160, 120)):
    """Imagen con estructura (ruido suavizado), para que el hash perceptual sea significativo."""
    rnd = random.Random(semilla)
    img = Image.new("RGB", (16, 12))
    img.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(16 * 12)])
    return img.resize(tam, Image.BICUBIC).filter(ImageFilter.GaussianBlur(2))


# ---------------------------------------------------------------- verificar_duplicados
def test_detecta_duplicados_y_variantes_pero_no_fotos_nuevas(tmp_path):
    ds, nuevas = tmp_path / "ds", tmp_path / "nuevas"
    (ds / "train").mkdir(parents=True)
    nuevas.mkdir()
    for i in range(12):
        foto_sintetica(i).save(ds / "train" / f"d{i}.jpg", quality=95)
    base = foto_sintetica(3)
    base.save(nuevas / "copia_exacta.jpg", quality=95)
    (nuevas / "copia_exacta.jpg").write_bytes((ds / "train" / "d3.jpg").read_bytes())
    base.resize((100, 75)).save(nuevas / "reducida.jpg", quality=70)
    ImageOps.mirror(base).save(nuevas / "volteada.jpg", quality=95)
    ImageEnhance.Brightness(base).enhance(1.25).save(nuevas / "mas_brillo.jpg", quality=95)
    for i in range(100, 104):
        foto_sintetica(i).save(nuevas / f"nueva{i}.jpg", quality=95)

    estado = {f["candidata"]: f["estado"] for f in vd.comparar(nuevas, ds)}
    assert estado["copia_exacta.jpg"] == "DUPLICADA EXACTA"
    assert all(estado[n].startswith("SOSPECHOSA") for n in ("reducida.jpg", "volteada.jpg", "mas_brillo.jpg")), estado
    assert all(estado[f"nueva{i}.jpg"] == "nueva" for i in range(100, 104)), estado


# ---------------------------------------------------------------- evaluar_externas
class _Plot(_Resultado):
    def plot(self):
        import numpy as np
        return np.zeros((20, 30, 3), dtype="uint8")


class ModeloFalso:
    names = NOMBRES

    def predict(self, ruta, conf, verbose):
        nombre = Path(ruta).stem
        if nombre.startswith("ok"):
            return [_Plot([0.9], [1])]
        if nombre.startswith("dudosa"):
            return [_Plot([0.5], [1])]
        if nombre.startswith("mala"):
            return [_Plot([0.85], [2])]  # Magdalena con alta confianza sobre una látigo
        return [_Plot([], [])]


def test_evaluar_externas_por_grupo(tmp_path):
    base = tmp_path / "ciegas"
    for grupo, fotos in {"en_agua": ["ok1", "ok2", "dudosa1"], "fuera_del_agua": ["mala1", "nada1"]}.items():
        d = base / grupo / "Hypanus_guttatus"
        d.mkdir(parents=True)
        for f in fotos:
            foto_sintetica(1, (64, 48)).save(d / f"{f}.jpg")
    (base / "en_agua" / "especie_rara").mkdir()
    filas = ee.evaluar(ModeloFalso(), base, tmp_path / "out")
    res = {r["grupo"]: r for r in ee.resumir(filas, "grupo")}
    assert res["en_agua"]["identificadas"] == 2 and res["en_agua"]["correctas"] == 2 and res["en_agua"]["no concluyentes"] == 1
    assert res["fuera_del_agua"]["errores con alta confianza"] == 1 and res["fuera_del_agua"]["no concluyentes"] == 1
    assert (tmp_path / "out" / "anotadas" / "en_agua" / "Hypanus_guttatus__ok1.jpg").exists()
    assert "| grupo |" in ee.tabla_md(ee.resumir(filas, "grupo"))


# ---------------------------------------------------------------- ingerir_validadas
def _png(color):
    b = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(b, "JPEG")
    return b.getvalue()


def test_ingesta_de_validadas_sin_modelo(tmp_path):
    (tmp_path / "img").mkdir()
    (tmp_path / "img" / "a.jpg").write_bytes(_png((1, 2, 3)))
    (tmp_path / "img" / "b.jpg").write_bytes(_png((9, 9, 9)))
    filas = [{"id_observacion": "u1", "archivo": "a.jpg", "especie_confirmada": "Hypanus guttatus", "validado_por": "Experta"},
             {"id_observacion": "u2", "archivo": "b.jpg", "especie_confirmada": "Pez inexistente", "validado_por": ""},
             {"id_observacion": "u3", "archivo": "no_existe.jpg", "especie_confirmada": "Hypanus_guttatus", "validado_por": ""}]
    r = iv.ingerir(filas, tmp_path / "img", tmp_path / "out")
    assert r["almacenadas"] == 1 and len(r["errores"]) == 2
    reg = json.loads((tmp_path / "out" / "metadata.jsonl").read_text().splitlines()[0])
    assert reg["especie_confirmada"] == "Hypanus_guttatus" and reg["class_id"] == 1 and reg["requiere_anotacion_manual"] is True
    # segunda ejecución: se detecta como duplicada
    assert iv.ingerir(filas[:1], tmp_path / "img", tmp_path / "out")["duplicadas"] == 1


def test_ingesta_con_modelo_propone_la_caja(tmp_path):
    (tmp_path / "img").mkdir()
    (tmp_path / "img" / "a.jpg").write_bytes(_png((5, 5, 5)))

    class M:
        names = NOMBRES

        def predict(self, img, conf, verbose):
            return [_Resultado([0.9], [1])]

    r = iv.ingerir([{"id_observacion": "u1", "archivo": "a.jpg", "especie_confirmada": "Aetobatus_narinari"}],
                   tmp_path / "img", tmp_path / "out", modelo=M())
    reg = json.loads((tmp_path / "out" / "metadata.jsonl").read_text().splitlines()[0])
    assert r["almacenadas"] == 1 and reg["caja_fuente"] == "modelo" and reg["class_id"] == 0


# ---------------------------------------------------------------- descargar_candidatas_inat
import descargar_candidatas_inat as dc  # noqa: E402


class _Resp:
    def __init__(self, data=None, contenido=b"jpg"):
        self._d, self.content = data, contenido

    def json(self):
        return self._d

    def raise_for_status(self):
        pass


class SesionFalsa:
    def __init__(self, paginas):
        self.paginas, self.llamadas = paginas, []

    def get(self, url, params=None, timeout=None):
        self.llamadas.append((url, params))
        if url.endswith("/observations"):
            i = params["page"] - 1
            return _Resp({"results": self.paginas[i] if i < len(self.paginas) else []})
        return _Resp(contenido=b"bytes-de-" + url.encode())


def _obs(i, user, lic="cc-by", fecha="2025-05-01"):
    return {"id": i, "user": {"login": user}, "observed_on": fecha, "place_guess": "Cartagena",
            "photos": [{"id": i * 10, "license_code": lic, "url": f"https://x/photos/{i}/square.jpg"}]}


def test_candidatas_filtra_excluidas_licencia_y_usuarios(monkeypatch):
    monkeypatch.setattr(dc.time, "sleep", lambda s: None)
    pag1 = [_obs(1, "ana"), _obs(2, "ana"), _obs(3, "ana"), _obs(4, "luis", lic=None), _obs(5, "luis", lic="cc-by-nc"),
            _obs(6, "eva", lic="all-rights-reserved")]
    pag2 = [_obs(7, "eva"), _obs(8, "marta")]
    s = SesionFalsa([pag1, pag2])
    r = dc.candidatas(s, "Hypanus guttatus", 10, excluidas={2}, place_id=7196)
    assert [x["observacion_id"] for x in r] == [1, 3, 5, 7, 8]   # 2 excluida; 3 es la 2.ª de ana; 4 y 6 sin licencia abierta
    assert r[0]["url_foto"] == "https://x/photos/1/large.jpg" and r[0]["url_observacion"].endswith("/observations/1")
    assert s.llamadas[0][1]["place_id"] == 7196 and s.llamadas[0][1]["quality_grade"] == "research"


def test_candidatas_se_detiene_al_llegar_a_la_cantidad(monkeypatch):
    monkeypatch.setattr(dc.time, "sleep", lambda s: None)
    s = SesionFalsa([[_obs(i, f"u{i}") for i in range(1, 30)]])
    assert len(dc.candidatas(s, "Aetobatus narinari", 5)) == 5


def test_descarga_guarda_imagenes_y_procedencia(tmp_path, monkeypatch):
    monkeypatch.setattr(dc.time, "sleep", lambda s: None)
    s = SesionFalsa([[_obs(11, "ana"), _obs(12, "luis")]])
    items = dc.descargar(s, dc.candidatas(s, "Styracura schmardae", 2), tmp_path)
    dc.guardar_procedencia(items, tmp_path)
    assert (tmp_path / "Styracura_schmardae" / "11_110.jpg").read_bytes().startswith(b"bytes-de-")
    filas = list(csv.DictReader(open(tmp_path / "procedencia.csv", encoding="utf-8")))
    assert [f["autor"] for f in filas] == ["ana", "luis"] and filas[0]["licencia"] == "cc-by"


def test_ids_excluidos_acepta_ids_y_urls(tmp_path):
    f = tmp_path / "usadas.txt"
    f.write_text("12345678\nhttps://www.inaturalist.org/observations/87654321\n\ntexto sin id\n")
    assert dc.ids_excluidos(f) == {12345678, 87654321}
