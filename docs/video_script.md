# Guion del video de demostración

Duración objetivo: entre 60 y 120 segundos. El video puede grabarse con la
aplicación ejecutándose en `http://127.0.0.1:5000`.

## Recorrido sugerido

| Tiempo | Acción en pantalla | Narración sugerida |
|---|---|---|
| 0:00–0:10 | Mostrar la pantalla inicial y los dos selectores. | “Esta es IrradianceKernel, una aplicación para comparar modelos de clasificación de irradiancia entrenados con datos Landsat y MODIS.” |
| 0:10–0:25 | Señalar los selectores A y B y pulsar **Comparar modelos**. | “Los modelos se cargan desde la carpeta `models`; la aplicación no entrena modelos desde la interfaz. Aquí selecciono los dos artefactos finales.” |
| 0:25–0:45 | Mostrar los mapas lado a lado y cambiar una capa base. | “Cada mapa conserva las observaciones de su satélite. Los colores representan la clase predicha y puedo cambiar entre calles, imagen satelital y OpenStreetMap.” |
| 0:45–1:00 | Desplazarse hasta las tarjetas de métricas. | “La comparación usa el mismo holdout 80/20 y muestra Accuracy, F1 macro, AUC OVR, MCC, matrices de confusión y la curva ROC o precisión-recall.” |
| 1:00–1:20 | Hacer clic sobre un punto del mapa y mostrar la sección de consulta. | “Al hacer clic, se busca la observación real más cercana y se muestran sus coordenadas, distancia, irradiancia registrada y la clase predicha por cada modelo.” |
| 1:20–1:30 | Mostrar brevemente el README o cerrar con la pantalla completa. | “El notebook documenta los puntos 1 a 5 y esta aplicación implementa el despliegue del punto 6.” |

## Recomendaciones de grabación

- Usar una ventana de navegador de al menos 1280 píxeles de ancho.
- Esperar a que aparezcan los puntos y las métricas antes de iniciar la
  narración.
- Mantener el cursor visible cuando se haga clic en el mapa.
- No mostrar rutas personales, tokens ni información del entorno.
- Exportar el video en formato MP4 y comprobar que dure entre 1 y 2 minutos.

## Enlace para completar la entrega

Después de subir el video, añadir aquí el enlace y copiarlo también en el
README raíz:

`Video de YouTube: PENDIENTE`
