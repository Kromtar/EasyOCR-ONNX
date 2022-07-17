# EasyOCR fork to export and use ONNX models

## Unofficial
This repository is not an official EasyOCR team repository.

## About the export tool
The tools for exporting EasyOCR models to ONNX only work if you have access to a CUDA-accelerated Nvidia graphics card. We must have the corresponding drivers in **CUDA version 11** and working in coordination with Docker.

Alternatively, you can download the models already exported from this [Drive folder](https://drive.google.com/drive/folders/1n_LOrJHkMVcZhyCgg37PYMAcsJ7_Sxsn?usp=sharing). After downloading the models corresponding to the language you want to use, you have to leave the files in `easyocr/onnx_models` using the names: `detection_model.onnx` and `recognition_model.onnx` accordingly.

Model export assumes: 
- That only 1 batch will be worked with. Tests have not yet been performed in order to have models that support multiple barch.
- That the height and width dimensions of the images to be analyzed are dynamic. It is possible to optimize the models by exporting them with fixed dimensions. This requires modification of the code.

## Branches of this respositiory
- **master**: Unmodified version of EasyOCR v1.5, Unmodified version of EasyOCR 1.5.
- **easyocr_hybrid**: Slightly modified version of EasyOCR to be able to export and use ONNX models. Uses PyTorch to preprocess and postprocess model input and output data. It is possible to deactivate ONNX and use EasyOCR as normal.
- **easyocr_only_onnx**: TODO: Highly modified version of EasyOCR that completely eliminates the use of PyTorch.

## Requirements
- Docker
- Docker Compose
- Nvidia card with CUDA 11 and their drivers. 

In case of not complying with the CUDA requirement, it is necessary to remove from line 11 to 17 of docker-compose.yml. In this case it will not be possible to export models to ONNX.

To ensure that you have CUDA correctly installed use the command: `nvidia-smi`. If a table is displayed where in the first line on the left it says "CUDA Version: 11.X", everything is correct. If not, check that you have installed the CUDA drivers correctly.

## Installation and Use

### Build and Run the Docker container
1. `docker-compose build`
2. `docker-compose run`

### Interacting

1. Enter to the container terminal: `docker exec -ti easyocr /bin/bash`

- Run `python3 onnx_export.py` to export models. The exported models will be stored in: `easyocr/onnx_models`. Now the method *readtext* has the *onnx_export* (bool) parameter to enable or disable the export of the ONNX model. Exporting will fail if you do not have CUDA.

- Run `python3 easyocr_with_onnx_model.py` to use EasyOCR with ONNX models. Now the method *readtext* has the *onnx* (bool) parameter to enable or disable the use of ONNX models.

## Additional comments

- [Poetry](https://python-poetry.org/) is being used for Python package installation.
- The system does not support exporting and/or using models for different language sets simultaneously.
