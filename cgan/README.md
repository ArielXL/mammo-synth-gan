# Tesis

## Instalar el enviroment y las dependencias del proyecto

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

## Documentación de los commits

Nos gustaría tener algunos estándares de nomenclatura en las confirmaciones. Vamos a intentar tanto como podamos usar el siguiente formato al nombrar nuestras confirmaciones:

```bash
<label>: <brief explanation>

<Optional body to explain your changes in more detail>
```

A partir de ahora, las etiquetas válidas son:

* `doc` for documentation related contributions.
* `fix` if you want to fix a bug.
* `feat` if you want to add any new feature.
* `ref` if you want to refactor some parts of the existing codebase.
* `test` if the change is related to adding tests to our project.
* `mig` for to run some migration.

### Ejemplos

```text
feat: Add user model, serializer, view, url endpoint, filter class and admin register
test: Add tests for user
mig: Create migrations for user
```

## Formatear el código

Estamos tratando de hacer que el código sea lo más estandarizado posible. Una forma de lograr esto es formateando su código correctamente. Se recomienda utilizar el formateador `black` después de haber realizado todos los cambios.

Puede formatear todo el proyecto ejecutando este comando en el directorio raíz:

```bash
make black
```

## Importaciones

El orden de las bibliotecas importadas es importante en Python. Entre otras cosas, facilita la identificación de bibliotecas de terceros para incluirlas en nuestros archivos de requisitos. Se recomienda usar `isort` para ordenar las bibliotecas correctamente. Se puede hacer ejecutando este comando en el directorio raíz:

```bash
sudo apt-get install isort
isort .
```
