# Fuentes y atribución

Los datasets, `KSVM.py`, `KANN.py` y `KernelUtilities.py` proceden del repositorio público [magohector/fkernel](https://github.com/magohector/fkernel/tree/master/Experimentos). Las copias descargadas se conservan sin modificar en `data/raw/` y `references/original/`.

La implementación de `src/irradiance_kernel/` es una adaptación educativa para APIs actuales de scikit-learn. Vectoriza los kernels, evita mutar parámetros durante `fit`, permite clonar y serializar estimadores, y añade `KRidgeClassifier`.

Referencia metodológica: D.-M. Pachajoa, H.-A. Mora-Paz y D. Mayorca-Torres, “Comparison of Kernel Functions in the Classification of Irradiance Zones from Multispectral Satellite Images”, Revista Facultad de Ingeniería, vol. 30, núm. 58, 2021. DOI: 10.19053/01211129.v30.n58.2021.13845.
