Fork para poder usar ONNX con EasyOCR
No oficial

Ramas:
- onnx-model-export: para poder exportar y usar los modelos de EsayOCR con ONNX
- XXXX rama para usar ONNX sin ninguna dependencia de torch

Requerimientos:
- Docker
- Docker Compose

Uso:

Haciendo la build de Docker:

1. "docker build"
2. "docker run"

Descargando la build:

1. Comando para levantar contenedor desde el Hub

Interactuado:

Exportando modelos:
1. Entrar a shell del contendor para poder interactuar con los scripts
2. python export.py
3. Selecciona el idioma del modelo que quieres exportar
4. Los modelos seran generados y guardados en XXX

Usando modelos con ONNX:

