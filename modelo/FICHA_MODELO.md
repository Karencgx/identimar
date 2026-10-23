# Ficha del modelo — identimar-rayas-yolov8n

- **Congelado:** 2026-10-14 23:46 UTC
- **Arquitectura / pesos de partida:** yolov8n.pt (transferencia de aprendizaje desde COCO)
- **Pesos:** `pesos/best.pt` · 6.26 MB · sha256 `d35ef9c336694abc828d85aefd4e8c8b87fdb213b4527c429d70e398c3abfb92`
- **Mejor época:** 93 de 100 ejecutadas · seed = 0
- **Dataset:** carpeta `IdentiMar-Rayas-2` (versión de Roboflow: 2)
- **Clases:** 0=Aetobatus_narinari, 1=Hypanus_guttatus, 2=Potamotrygon_magdalenae, 3=Styracura_schmardae
- **Contrato:** confianza < 0.7 → `no_concluyente`

## Composición del dataset (auditoría)
- **train**: 876 imágenes; por clase: Aetobatus_narinari=168, Hypanus_guttatus=168, Potamotrygon_magdalenae=372, Styracura_schmardae=168; copias `_dup`: 279
- **valid**: 58 imágenes; por clase: Aetobatus_narinari=16, Hypanus_guttatus=17, Potamotrygon_magdalenae=9, Styracura_schmardae=16; copias `_dup`: 0
- **test**: 29 imágenes; por clase: Aetobatus_narinari=8, Hypanus_guttatus=8, Potamotrygon_magdalenae=5, Styracura_schmardae=8; copias `_dup`: 0

## Hiperparámetros
```yaml
amp: true
batch: 16
close_mosaic: 10
data: /content/IdentiMar-Rayas-2/data.yaml
degrees: 0.0
deterministic: true
epochs: 100
fliplr: 0.5
flipud: 0.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
imgsz: 640
lr0: 0.01
lrf: 0.01
mixup: 0.0
model: yolov8n.pt
momentum: 0.937
mosaic: 1.0
optimizer: auto
patience: 20
pretrained: true
scale: 0.5
seed: 0
shear: 0.0
translate: 0.1
warmup_epochs: 3.0
weight_decay: 0.0005
```

## Métricas finales (P, R y F1 al umbral de máximo F1 de Ultralytics)
### test
| Clase | Imágenes | Instancias | Precisión | Recall | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| Aetobatus_narinari | 8 | 8 | 0.741 | 0.875 | 0.802 | 0.875 | 0.722 |
| Hypanus_guttatus | 8 | 8 | 0.847 | 1 | 0.917 | 0.967 | 0.749 |
| Potamotrygon_magdalenae | 5 | 5 | 0.825 | 0.948 | 0.882 | 0.962 | 0.693 |
| Styracura_schmardae | 8 | 8 | 0.978 | 1 | 0.989 | 0.995 | 0.87 |
| Promedio (macro) | — | 29 | 0.847 | 0.956 | 0.897 | 0.95 | 0.759 |

### valid
| Clase | Imágenes | Instancias | Precisión | Recall | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| Aetobatus_narinari | 16 | 16 | 0.939 | 0.969 | 0.954 | 0.984 | 0.706 |
| Hypanus_guttatus | 17 | 18 | 0.866 | 0.72 | 0.786 | 0.835 | 0.545 |
| Potamotrygon_magdalenae | 9 | 9 | 0.679 | 0.556 | 0.611 | 0.566 | 0.4 |
| Styracura_schmardae | 16 | 16 | 0.876 | 0.875 | 0.876 | 0.95 | 0.739 |
| Promedio (macro) | — | 59 | 0.84 | 0.78 | 0.807 | 0.834 | 0.598 |

## Entorno
```json
{
  "python": "3.13.15",
  "so": "Linux-6.6.122+-x86_64-with-glibc2.39",
  "ultralytics": "8.4.170",
  "torch": "2.11.0+cu130",
  "roboflow": "1.7.1",
  "numpy": "2.1.3"
}
```

## Completar a mano
- [ ] Enlace al repositorio y commit/tag de este modelo: 
- [ ] Proyecto y versión de Roboflow (confirmar): 
- [ ] Sobremuestreo aplicado (clase, copias por imagen, imágenes de origen): 
- [ ] Responsable de la corrida: 
