"""Catálogo de especies que reconoce el modelo (para armonizar la taxonomía con el backend y la app)."""

NOMBRES_COMUNES = {
    "Hypanus_guttatus": "Raya látigo",
    "Aetobatus_narinari": "Raya gavilán",
    "Styracura_schmardae": "Raya chupare",
    "Potamotrygon_magdalenae": "Raya del Magdalena",
}


def catalogo(names: dict) -> list:
    """names: {id_de_clase: nombre_de_clase} tal como lo entrega el modelo."""
    return [{"id": int(i), "clase": n, "nombre_cientifico": n.replace("_", " "), "nombre_comun": NOMBRES_COMUNES.get(n)}
            for i, n in sorted(names.items())]
