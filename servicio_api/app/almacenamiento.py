"""Descarga de imágenes desde una referencia de almacenamiento (URL).

Es opcional y viene DESHABILITADA: solo funciona si se configura ALLOWED_IMAGE_HOSTS.
Medidas contra SSRF (que el servicio sea usado para llamar a recursos internos):
solo https, solo hosts de la lista, sin seguir redirecciones, con límite de tamaño y de tiempo.
"""
from urllib.parse import urlparse

import httpx


def descargar_imagen(url: str, hosts_permitidos: list, max_bytes: int, timeout: float) -> bytes:
    partes = urlparse(url)
    if partes.scheme != "https":
        raise ValueError("Solo se permiten URLs https.")
    if (partes.hostname or "").lower() not in hosts_permitidos:
        raise PermissionError("El host de la URL no está en la lista de almacenamientos permitidos.")

    trozos, total = [], 0
    with httpx.Client(timeout=timeout, follow_redirects=False) as cliente:
        with cliente.stream("GET", url) as r:
            if r.status_code != 200:
                raise IOError(f"El almacenamiento respondió {r.status_code}.")
            for trozo in r.iter_bytes():
                total += len(trozo)
                if total > max_bytes:
                    raise OverflowError("La imagen supera el tamaño máximo permitido.")
                trozos.append(trozo)
    return b"".join(trozos)
