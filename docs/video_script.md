# Guion para video (90 segundos)

## 0:00–0:15 — Propósito

“Esta aplicación compara pipelines con funciones kernel para clasificar zonas de irradiancia a partir de observaciones Landsat y MODIS en Nariño. Se evaluaron escaladores, cuatro discretizadores, PCA o LDA, tres modelos y nueve kernels.”

Mostrar brevemente el notebook y la tabla de resultados.

## 0:15–0:35 — Selección

Abrir la aplicación, elegir `LANDSAT` y señalar los tres modelos disponibles. Seleccionar dos configuraciones distintas y pulsar **Comparar modelos**.

“Los modelos se seleccionaron por F1 macro en validación cruzada. El holdout del 20% solo se utilizó para la evaluación final.”

## 0:35–1:00 — Mapas y consulta

Mostrar ambos mapas, sus colores de clase y clicar una ubicación.

“Cada punto es una observación real, no una superficie interpolada. Al hacer clic se busca el punto observado más cercano y se muestran su irradiancia, la clase predicha, la clase observada y si cada modelo acertó.”

## 1:00–1:20 — Métricas

Desplazarse a Accuracy, balanced accuracy, F1 macro, menor F1 de clase, MCC, AUC, F1 por clase, matrices de confusión y curvas.

“La comparación no depende únicamente de accuracy: usamos F1 macro como criterio principal y mostramos la clase más débil y una línea base ingenua. La validación espacial adicional comprueba cuánto baja el desempeño al dejar zonas completas fuera.”

## 1:20–1:30 — Cierre

“El repositorio incluye los datasets originales, el notebook reproducible, seis modelos almacenados, pruebas y las instrucciones para ejecutar la aplicación localmente.”

Añadir el enlace del video al README después de subirlo a YouTube.
