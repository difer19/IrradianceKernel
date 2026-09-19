# Irradiance Kernel Lab

Proyecto educativo para comparar funciones kernel en la clasificación de irradiancia usando observaciones Landsat y MODIS de Nariño, Colombia. Incluye un notebook explicado, 1.296 estructuras experimentales, seis modelos finales y una aplicación Flask de comparación geográfica.

## 1. Preparar el entorno

Este proyecto vive dentro del entorno virtual de la carpeta `ML`. Desde esa carpeta:

```bash
source bin/activate
cd IrradianceKernel
python --version  # debe mostrar Python 3.12.x
python -m pip install -r requirements.txt
```

También puede ejecutar sin activar el entorno usando `../bin/python`.

## 2. Recorrido didáctico

1. Abra `notebooks/irradiance_kernel_comparison.ipynb` y ejecute las secciones de contexto, datos y discretización.
2. Revise `src/irradiance_kernel/kernels.py`, donde las nueve funciones producen matrices Gram.
3. Revise `estimators.py`: `KSVC`, `KANNC` y el nuevo `KRidgeClassifier` comparten una API de clasificación.
4. Observe en `pipeline.py` que el discretizador se ajusta dentro de `fit`; por eso el fold de validación nunca define sus clases.
5. Ejecute la matriz completa y compare los resultados del notebook.
6. Inicie Flask y explore los modelos seleccionados.

## 3. Datos

Cada CSV contiene las columnas `latitude`, `longitude`, `band1`–`band7` y `value`. Aunque los encabezados dicen latitud/longitud, las dos primeras columnas son coordenadas proyectadas EPSG:3857 en metros. La aplicación las convierte a WGS84 para mostrarlas.

Las URLs, fechas y checksums se encuentran en `data/metadata/sources.json`. Los archivos originales no se reescriben. También se conserva el límite departamental usado por el notebook de ejemplo y se genera una copia GeoJSON en WGS84 con `scripts/convert_boundary.py`.

## 4. Ejecutar los experimentos

Diagnóstico corto:

```bash
../bin/python run_experiments.py --max-configs 12 --jobs 1
```

Matriz completa (648 configuraciones por dataset):

```bash
../bin/python run_experiments.py --jobs 4
```

Se usa un holdout fijo del 20% y KFold de tres particiones sobre el 80% restante. Toda transformación, incluida la discretización de la irradiancia, se ajusta dentro del fold. La selección usa F1 macro, seguida por MCC, AUC, accuracy y tiempo.

No se optimizan hiperparámetros. Los resultados comparan los parámetros fijos documentados y no deben interpretarse como el máximo rendimiento posible de cada algoritmo.

Los resultados quedan en `artifacts/*_cv_results.csv`; los tres ganadores por satélite, sus gráficas y el manifiesto quedan en `artifacts/models`, `artifacts/reports` y `artifacts/manifest.json`.

## 5. Ejecutar la aplicación

Después de generar los artefactos:

```bash
../bin/python app/app.py
```

Abra [http://127.0.0.1:5000](http://127.0.0.1:5000). Los selectores solo permiten comparar modelos del mismo satélite. Un clic sobre cualquier mapa consulta la observación más cercana; no se realiza interpolación espacial.

El contorno oficial de Nariño se dibuja únicamente como referencia geográfica. Los colores representan predicciones en observaciones reales, no una estimación continua entre puntos. Esta decisión sustituye el Kriging del notebook orientativo porque la salida de este proyecto es categórica.

API disponible:

- `GET /api/models`
- `GET /api/boundary`
- `GET /api/compare?model_a=...&model_b=...`
- `GET /api/point?satellite=...&lat=...&lon=...&model_a=...&model_b=...`

## 6. Verificación

```bash
../bin/python -m pytest
../bin/python -m compileall src app
../bin/jupyter nbconvert --to notebook --execute notebooks/irradiance_kernel_comparison.ipynb --output /tmp/irradiance-verified.ipynb --ExecutePreprocessor.timeout=600
```

## 7. Estructura

```text
IrradianceKernel/
├── app/                 # Flask, interfaz y API
├── artifacts/           # Resultados, modelos y figuras
├── data/                # CSV, límite geográfico y metadatos
├── docs/                # Atribución y guion del video
├── notebooks/           # Entregable de los puntos 1–4
├── references/original/ # Código original requerido
├── src/                 # Implementación reproducible
└── tests/               # Pruebas automatizadas
```

## 8. Video y publicación

El guion de 90 segundos está en `docs/video_script.md`.

- Video de YouTube: _añadir enlace después de grabarlo_.
- Repositorio GitHub: _añadir URL después de publicarlo_.

El proyecto se distribuye bajo GPL-3.0; consulte `LICENSE` y `docs/ATTRIBUTION.md` antes de redistribuirlo.
