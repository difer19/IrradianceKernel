# Fuentes y atribución

Los datasets, `KSVM.py`, `KANN.py`, `KernelUtilities.py` y el shapefile de Nariño proceden del repositorio público [magohector/fkernel](https://github.com/magohector/fkernel/tree/master/Experimentos). Las copias de datos y código de referencia se conservan sin modificar en `data/raw/`, `data/geography/` y `references/original/`. El archivo `narino_boundary.geojson` es una transformación reproducible del shapefile EPSG:3857 a WGS84 mediante `scripts/convert_boundary.py`.

La implementación de `src/irradiance_kernel/` es una adaptación educativa para APIs actuales de scikit-learn. Vectoriza los kernels, evita mutar parámetros durante `fit`, permite clonar y serializar estimadores, y añade `KRidgeClassifier`.

Referencia metodológica: D.-M. Pachajoa, H.-A. Mora-Paz y D. Mayorca-Torres, “Comparison of Kernel Functions in the Classification of Irradiance Zones from Multispectral Satellite Images”, Revista Facultad de Ingeniería, vol. 30, núm. 58, 2021. DOI: 10.19053/01211129.v30.n58.2021.13845.

El código original declara sus implementaciones bajo GNU GPL. Para preservar compatibilidad con ese material y dejar explícitas las condiciones de redistribución, este proyecto se publica bajo GNU General Public License v3.0; el texto íntegro se encuentra en `LICENSE`.
