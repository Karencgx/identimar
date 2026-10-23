"""
IdentiMar - Grupo 9 (Visión Artificial)
Salida estándar del módulo de visión artificial, según el contrato con el Equipo 3 (backend).

Formato de respuesta (JSON):
    id_observacion    : str
    especie_propuesta : str | None   (None si el modelo no detectó nada)
    confianza         : float
    umbral_minimo     : float
    estado            : "identificado" | "no_concluyente"

Este archivo NO depende de torch ni de ultralytics a propósito: así lo pueden importar
tal cual desde la API (FastAPI) en la siguiente etapa.
"""

UMBRAL_POR_DEFECTO = 0.70


def generar_respuesta(resultado, id_observacion, umbral=UMBRAL_POR_DEFECTO):
    """
    Convierte el resultado de `model.predict(...)[0]` (Ultralytics) en la respuesta del contrato.

    Regla: se toma la detección de MAYOR confianza como especie propuesta.
    - sin detecciones                       -> no_concluyente, especie None, confianza 0.0
    - confianza >= umbral                   -> identificado
    - confianza <  umbral                   -> no_concluyente (el backend lo deriva a revisión experta)

    IMPORTANTE: llamen a predict con un `conf` bajo (ej. 0.25), NO con 0.70. Así, cuando la
    confianza queda entre 0.25 y 0.70 el sistema igual informa la especie que sospecha y
    marca el caso como no_concluyente, en vez de devolver "nada".
    """
    umbral = float(umbral)
    confs = resultado.boxes.conf.tolist() if len(resultado.boxes) else []
    clases = resultado.boxes.cls.tolist() if len(resultado.boxes) else []

    if not confs:
        return {
            "id_observacion": str(id_observacion),
            "especie_propuesta": None,
            "confianza": 0.0,
            "umbral_minimo": umbral,
            "estado": "no_concluyente",
        }

    idx = max(range(len(confs)), key=confs.__getitem__)
    confianza = float(confs[idx])
    especie = resultado.names[int(clases[idx])]

    return {
        "id_observacion": str(id_observacion),
        "especie_propuesta": especie,
        "confianza": round(confianza, 4),
        "umbral_minimo": umbral,
        "estado": "identificado" if confianza >= umbral else "no_concluyente",
    }
