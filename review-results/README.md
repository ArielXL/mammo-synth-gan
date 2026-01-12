# Clasificación de los resultados

Este proyecto contiene scripts y datos para analizar resultados de experimentos con CGAN (Conditional Generative Adversarial Networks).

## Estructura del proyecto

``` bash
.
├── pairs.csv
├── README.md
├── requirements.txt
└── src/
	├── makefile
	└── review_results.py
```

## Descripción de archivos

- `pairs.csv`: Archivo de datos con pares utilizados en los experimentos.
- `requirements.txt`: Lista de dependencias necesarias para ejecutar los scripts.
- `src/review_results.py`: Script principal para analizar y visualizar los resultados.
- `src/makefile`: Automatización de tareas comunes (ejecución, limpieza, etc.).

## Instalación

1. Clona el repositorio.
2. Instala las dependencias:

   ```bash
   pip3 install -r requirements.txt
   ```

## Uso

Ejecuta el script principal desde la carpeta `src`:

```bash
python3 src/review_results.py
```

## Contribución

Las contribuciones son bienvenidas. Por favor, abre un issue o un pull request para sugerencias o mejoras.

## Licencia

Este proyecto se distribuye bajo la licencia CIC.
