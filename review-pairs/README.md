# review-pairs

Esta carpeta contiene los pares de imágenes y máscaras seleccionados para el entrenamiento y evaluación del modelo cGAN de síntesis mamográfica.

## Propósito

- Almacenar los pares de casos revisados y aceptados para el pipeline.
- Facilitar la organización y trazabilidad de los datos utilizados en el entrenamiento y test.
- Permitir la revisión manual y automática de los pares antes de ser procesados por el modelo.

## Estructura sugerida

- Cada subcarpeta debe corresponder a un caso o paciente, con las imágenes y máscaras asociadas.
- Los nombres de los archivos deben seguir el formato estándar del proyecto (`before_image.png`, `after_image.png`, `before_mask.png`, `after_mask.png`, etc.).
- Puede incluir archivos de metadatos o anotaciones para cada par.

## Uso en el pipeline

1. El script principal (`main.py`) lee los pares desde esta carpeta para construir los datasets de entrenamiento y test.
2. Solo se procesan los pares que han sido revisados y marcados como válidos.
3. Los resultados generados por el modelo pueden ser comparados con los pares originales para calcular métricas de calidad.

## Recomendaciones

- Mantén esta carpeta actualizada con los pares más recientes y validados.
- Realiza revisiones periódicas para asegurar la calidad de los datos.
- Documenta cualquier cambio o decisión sobre la inclusión/exclusión de pares.

---

Para dudas o sugerencias sobre la gestión de los pares, contacta al responsable del proyecto.
