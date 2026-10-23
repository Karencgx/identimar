"""Validaciones de entrada compartidas."""
import re

from fastapi import HTTPException

# Identificadores de observación: letras, números, punto, guion y guion bajo (UUIDs y enteros caben).
# También evita caracteres peligrosos en logs y en nombres de archivo.
_ID_VALIDO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def validar_id(valor: str) -> str:
    valor = (valor or "").strip()
    if not _ID_VALIDO.match(valor):
        raise HTTPException(
            status_code=400,
            detail="id_observacion inválido: use 1-64 caracteres entre letras, números, '.', '-' y '_'.",
        )
    return valor
