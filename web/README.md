# Aplicación web de irradiancia

Esta aplicación Flask carga los dos modelos finales que están en la carpeta
`../models` y los datasets originales que están en `../datasets`. No entrena
modelos desde la interfaz ni permite subir artefactos externos.

## Ejecución local

Con el entorno ya activado, desde la raíz de `IrradianceKernel` ejecuta:

```bash
cd web
flask run
```

Después abre <http://127.0.0.1:5000>.

## Funcionalidad

- Selector de dos modelos almacenados en `models/`.
- Mapas lado a lado con las observaciones Landsat y MODIS. Cada mapa inicia
  con un mapa cartográfico real de Esri y permite cambiar a imagen satelital
  de Esri u OpenStreetMap desde el control de capas; por ello se requiere
  conexión a internet para descargar las teselas base.
- Comparación de Accuracy, F1 macro, AUC OVR y MCC sobre el mismo holdout
  80/20 con semilla 2021 del notebook.
- Matrices de confusión y curva ROC o precision-recall según el balance de
  clases.
- Clic sobre el mapa para localizar la observación más cercana y mostrar su
  irradiancia registrada y las clases predichas.

La API disponible es:

```text
GET /api/models
GET /api/compare?model_a=landsat_best&model_b=modis_best
GET /api/point?satellite=landsat&lat=...&lon=...&model_a=...&model_b=...
```

Las coordenadas de los CSV están en EPSG:3857; en esos archivos, por cómo se
generaron, la columna `latitude` contiene el eje X y `longitude` el eje Y. La
interfaz corrige esa correspondencia y las convierte a latitud/longitud para
representarlas en Leaflet. Cuando se comparan Landsat y
MODIS, cada mapa conserva sus observaciones nativas. Una predicción cruzada en
el detalle del clic se marca como exploratoria.
