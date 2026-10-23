"""Decodificación y validación de las imágenes que llegan al servicio."""
import io

from PIL import Image, ImageOps, UnidentifiedImageError

FORMATOS_PERMITIDOS = {"JPEG", "PNG", "WEBP"}
LADO_MINIMO = 32  # px


class ImagenInvalida(Exception):
    pass


class ImagenMuyGrande(Exception):
    pass


def decodificar_imagen(datos: bytes, max_pixels: int) -> Image.Image:
    """Devuelve una imagen PIL en RGB y con la orientación EXIF aplicada.

    La orientación importa: las fotos de celular suelen guardarse "acostadas" con una etiqueta EXIF.
    El dataset de entrenamiento se auto-orientó en Roboflow, así que en inferencia hay que hacer lo mismo.
    """
    try:
        img = Image.open(io.BytesIO(datos))
    except Image.DecompressionBombError:
        raise ImagenMuyGrande("La imagen tiene demasiados píxeles.")
    except (UnidentifiedImageError, OSError):
        raise ImagenInvalida("El archivo no es una imagen válida (use JPEG, PNG o WEBP).")

    if img.format not in FORMATOS_PERMITIDOS:
        raise ImagenInvalida(f"Formato no soportado: {img.format}. Use JPEG, PNG o WEBP.")

    ancho, alto = img.size  # no decodifica todavía
    if ancho * alto > max_pixels:
        raise ImagenMuyGrande(f"La imagen tiene {ancho}x{alto} px; el máximo es {max_pixels:,} píxeles.")
    if min(ancho, alto) < LADO_MINIMO:
        raise ImagenInvalida(f"La imagen es demasiado pequeña ({ancho}x{alto}); mínimo {LADO_MINIMO} px por lado.")

    try:
        img.load()
    except Exception:
        raise ImagenInvalida("La imagen está corrupta o incompleta.")

    return ImageOps.exif_transpose(img).convert("RGB")
