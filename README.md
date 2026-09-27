# IrradianceKernel

Clasificación de irradiancia con funciones kernel usando observaciones de los
satélites Landsat y MODIS. El proyecto contiene el notebook de los puntos 1–5
y una aplicación Flask para el punto 6.

## Entregables

- **Puntos 1–5:** `notebook/Classification/Clasification.ipynb`.
  Cada bloque comienza con un encabezado Markdown que identifica el punto del
  enunciado: carga de datos, clases originales `KSVC`/`KANNC`,
  `KRidgeClassifier`, campo de pipelines y evaluación.
- **Punto 6:** `web/`, junto con los modelos finales de `models/`.
- **Video:** guion de demostración de 1–2 minutos en
  [`docs/video_script.md`](docs/video_script.md). El enlace de YouTube se debe
  añadir después de grabar el video.

## Estructura

```text
IrradianceKernel/
├── datasets/                         CSV originales Landsat y MODIS
├── models/                           artefactos joblib usados por la web
├── notebook/Classification/           notebook de los puntos 1–5
├── notebook/Regression/               notebook orientativo existente
├── web/                               aplicación Flask del punto 6
├── docs/video_script.md               guion para el video
├── requirements.txt                   dependencias reproducibles
└── README.md
```

## Requisitos

- Python 3.12.
- El entorno del curso disponible en `../bin/python` cuando se ejecuta desde
  la carpeta `IrradianceKernel`.
- Conexión a internet para mostrar las teselas de Esri u OpenStreetMap en el
  mapa. Los modelos y los datos se cargan localmente.

Instalación desde la raíz del proyecto:

```bash
cd IrradianceKernel
../bin/python -m pip install -r requirements.txt
```

Si se utiliza otro entorno virtual, se puede sustituir `../bin/python` por el
ejecutable de ese entorno.

## Ejecutar el notebook

Abrir Jupyter desde `IrradianceKernel`:

```bash
../bin/python -m jupyter lab
```

Después abrir `notebook/Classification/Clasification.ipynb` y ejecutar las
celdas en orden. La celda de evaluación recorre el campo experimental completo
(648 configuraciones por satélite); puede tardar varios minutos. La semilla
2021, el holdout 80/20 y los índices de los folds están fijados para que la
comparación sea reproducible.

También se puede generar una copia ejecutada con `nbconvert`:

```bash
../bin/python -m jupyter nbconvert \
  --to notebook --execute notebook/Classification/Clasification.ipynb \
  --output Clasification_ejecutado.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3
```

La última sección del notebook reentrena los ganadores con el 80 % de cada
dataset y guarda `models/landsat_best.joblib` y `models/modis_best.joblib`.

## Ejecutar en Google Colab

Se puede abrir directamente desde
[Google Colab](https://colab.research.google.com/github/difer19/IrradianceKernel/blob/main/notebook/Classification/Clasification.ipynb).
La primera sección permite indicar manualmente la carpeta de Google Drive que
contiene los dos CSV. Por ejemplo:

```python
from google.colab import drive
drive.mount('/content/drive')
DATASET_DIR = '/content/drive/MyDrive/IrradianceKernel/datasets'
```

La ruta debe contener exactamente `landsat_model.csv` y `modis_model.csv`.
Después se ejecutan las celdas en orden.

Si Colab informa que falta alguna dependencia, ejecutar una vez en una celda:

```python
%pip install -q pandas numpy scikit-learn matplotlib joblib
```

Después se ejecutan las celdas en orden. La evaluación completa (`RUN_FULL_GRID
= True`) recorre 648 configuraciones por satélite y puede tardar; para una
prueba rápida se puede cambiar temporalmente a `False` en la celda de
evaluación y volver a dejarlo en `True` para la entrega final.

## Ejecutar la aplicación web

Desde la raíz de `IrradianceKernel`:

```bash
../bin/flask --app web/app.py run --debug
```

Abrir <http://127.0.0.1:5000> en el navegador. La aplicación no entrena ni
acepta modelos externos: carga exclusivamente los dos archivos de `models/`.

La interfaz permite:

1. Seleccionar dos modelos y comparar Accuracy, F1 macro, AUC OVR y MCC.
2. Ver los puntos clasificados en mapas lado a lado.
3. Cambiar la capa base entre calles de Esri, imagen satelital de Esri y
   OpenStreetMap.
4. Consultar con un clic el punto observado más cercano, su irradiancia y la
   clase predicha por ambos modelos.
5. Revisar matrices de confusión y la curva ROC o precisión-recall.

API disponible:

```text
GET /api/models
GET /api/compare?model_a=landsat_best&model_b=modis_best
GET /api/point?satellite=landsat&lat=...&lon=...&model_a=...&model_b=...
```

## Video de demostración

El guion para el video de 1–2 minutos está en
[`docs/video_script.md`](docs/video_script.md). Después de grabarlo, añadir el
enlace de YouTube aquí:

> **Video de demostración:** [pendiente de añadir enlace de YouTube]

## Fuentes y atribución

- Datasets: [landsat_model.csv](https://github.com/magohector/fkernel/blob/master/Experimentos/landsat_model.csv) y
  [modis_model.csv](https://github.com/magohector/fkernel/blob/master/Experimentos/modis_model.csv).
- Clases originales: [KSVM.py](https://github.com/magohector/fkernel/blob/master/Experimentos/KSVM.py) y
  [KANN.py](https://github.com/magohector/fkernel/blob/master/Experimentos/KANN.py).
- Capas cartográficas: Esri y OpenStreetMap, con atribución visible en cada
  mapa.
