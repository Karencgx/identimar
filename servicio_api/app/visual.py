"""Dibuja las detecciones sobre la imagen (para la demostración y la depuración)."""
from PIL import Image, ImageDraw, ImageFont

COLORES = ["#1f8a8f", "#e07b4c", "#6b5bd2", "#b8932a"]


def anotar(imagen: Image.Image, resultado, nombres: dict) -> Image.Image:
    img = imagen.copy()
    d = ImageDraw.Draw(img)
    lado = max(img.size)
    grosor, tam = max(2, round(lado / 300)), max(14, round(lado / 45))
    fuente = ImageFont.load_default(size=tam)
    if len(resultado.boxes) == 0:
        return img
    cajas, confs, clases = (resultado.boxes.xyxy.tolist(), resultado.boxes.conf.tolist(), resultado.boxes.cls.tolist())
    for (x1, y1, x2, y2), conf, c in zip(cajas, confs, clases):
        color = COLORES[int(c) % len(COLORES)]
        d.rectangle([x1, y1, x2, y2], outline=color, width=grosor)
        texto = f"{nombres.get(int(c), int(c))} {conf:.2f}"
        ancho = d.textlength(texto, font=fuente)
        ty = y1 - tam - 8 if y1 - tam - 8 >= 0 else y1 + 2
        d.rectangle([x1, ty, x1 + ancho + 10, ty + tam + 6], fill=color)
        d.text((x1 + 5, ty + 2), texto, fill="white", font=fuente)
    return img
