"""Descarga fotos CANDIDATAS de iNaturalist, con su procedencia, para armar el conjunto ciego (imágenes nuevas).

    pip install requests
    python scripts/descargar_candidatas_inat.py --especie "Hypanus guttatus" --cantidad 12 --salida ./candidatas \
        [--excluir ids_ya_usados.txt] [--solo-colombia]

- Solo observaciones de calidad "research" y fotos con licencia abierta (CC0, CC-BY, CC-BY-NC, CC-BY-SA, CC-BY-NC-SA).
- --excluir: archivo de texto con las observaciones que YA usaron (un ID o URL por línea; salen de la columna
  URL_o_referencia de su inventario). Evita repetir una observación, pero NO detecta la misma foto subida dos veces:
  pasen siempre después scripts/verificar_duplicados.py contra el dataset.
- Máximo 2 fotos por observador, para que el conjunto sea variado.
- Deja <salida>/<especie>/<obs>_<foto>.jpg y <salida>/procedencia.csv (fuente, licencia, autor, fecha, lugar).
"""
import argparse
import csv
import re
import sys
import time
from pathlib import Path

API = "https://api.inaturalist.org/v1"
LICENCIAS = {"cc0", "cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa"}


def ids_excluidos(ruta):
    ids = set()
    if ruta:
        for linea in Path(ruta).read_text(encoding="utf-8").splitlines():
            m = re.search(r"(\d{4,})\s*$", linea.strip())
            if m:
                ids.add(int(m.group(1)))
    return ids


def candidatas(sesion, especie, cantidad, excluidas=frozenset(), licencias=LICENCIAS, place_id=None,
               max_por_usuario=2, calidad="research", espera=1.0, max_paginas=15):
    """Devuelve hasta `cantidad` dicts con la observación y la foto elegida (sin descargar nada)."""
    elegidas, por_usuario = [], {}
    for pagina in range(1, max_paginas + 1):
        params = {"taxon_name": especie, "photos": "true", "quality_grade": calidad, "per_page": 100,
                  "page": pagina, "order_by": "created_at", "order": "desc"}
        if place_id:
            params["place_id"] = place_id
        r = sesion.get(f"{API}/observations", params=params, timeout=30)
        r.raise_for_status()
        resultados = r.json().get("results", [])
        if not resultados:
            break
        for obs in resultados:
            usuario = (obs.get("user") or {}).get("login", "")
            if obs["id"] in excluidas or por_usuario.get(usuario, 0) >= max_por_usuario:
                continue
            foto = next((f for f in obs.get("photos", []) if (f.get("license_code") or "").lower() in licencias), None)
            if foto is None:
                continue
            por_usuario[usuario] = por_usuario.get(usuario, 0) + 1
            elegidas.append({"especie": especie, "observacion_id": obs["id"], "foto_id": foto["id"],
                             "url_observacion": f"https://www.inaturalist.org/observations/{obs['id']}",
                             "url_foto": re.sub(r"/(square|small|medium)\.", "/large.", foto["url"]),
                             "autor": usuario, "licencia": foto["license_code"], "fecha": obs.get("observed_on") or "",
                             "lugar": obs.get("place_guess") or ""})
            if len(elegidas) >= cantidad:
                return elegidas
        time.sleep(espera)
    return elegidas


def descargar(sesion, items, salida):
    salida = Path(salida)
    for it in items:
        destino = salida / it["especie"].replace(" ", "_") / f"{it['observacion_id']}_{it['foto_id']}.jpg"
        destino.parent.mkdir(parents=True, exist_ok=True)
        r = sesion.get(it["url_foto"], timeout=60)
        r.raise_for_status()
        destino.write_bytes(r.content)
        it["archivo"] = str(destino)
    return items


def guardar_procedencia(items, salida):
    ruta = Path(salida) / "procedencia.csv"
    nuevo = not ruta.exists()
    cols = ["especie", "observacion_id", "foto_id", "url_observacion", "url_foto", "autor", "licencia", "fecha", "lugar", "archivo"]
    with open(ruta, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if nuevo:
            w.writeheader()
        w.writerows(items)


def main():
    import requests

    ap = argparse.ArgumentParser()
    ap.add_argument("--especie", required=True, help='nombre científico, p. ej. "Hypanus guttatus"')
    ap.add_argument("--cantidad", type=int, default=10)
    ap.add_argument("--salida", default="candidatas")
    ap.add_argument("--excluir")
    ap.add_argument("--solo-colombia", action="store_true")
    a = ap.parse_args()
    s = requests.Session()
    s.headers["User-Agent"] = "IdentiMar-Grupo9 (proyecto universitario UdeA)"
    place = None
    if a.solo_colombia:
        res = s.get(f"{API}/places/autocomplete", params={"q": "Colombia"}, timeout=30).json().get("results", [])
        place = next((p["id"] for p in res if p.get("name", "").lower() == "colombia"), None)
    items = candidatas(s, a.especie, a.cantidad, ids_excluidos(a.excluir), place_id=place)
    if not items:
        sys.exit("No hay candidatas nuevas para esa especie con licencia abierta. Busquen otras fuentes (SiB Colombia, "
                 "tesis, fotos propias o de pescadores).")
    descargar(s, items, a.salida)
    guardar_procedencia(items, a.salida)
    print(f"{len(items)} candidatas guardadas en {a.salida}. Ahora ejecuten scripts/verificar_duplicados.py.")


if __name__ == "__main__":
    main()
