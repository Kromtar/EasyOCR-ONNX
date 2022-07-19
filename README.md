# EasyOCR fork to export and use ONNX models

## Unofficial
This repository is not an official EasyOCR team repository.

This repository contains the environments and test scripts to export and use the EasyOCR models in ONNX format. A minor modification of EasyOCR is presented, where ONNX has been integrated (*easyocr_hybrid_onnx*). And another major modification of EasyOCR, where all PyTorch dependencies have been removed (*easyocr_only_onnx*).

### Detail about the export tool in easyocr_hybrid_onnx
The tools for exporting EasyOCR models to ONNX only work if you have access to a CUDA-accelerated Nvidia graphics card. We must have the corresponding drivers in **CUDA version 11** installed and working in coordination with Docker.

Alternatively, you can download the models, already exported, from this [Drive folder](https://drive.google.com/drive/folders/1n_LOrJHkMVcZhyCgg37PYMAcsJ7_Sxsn?usp=sharing). After downloading the models corresponding to the language you want to use, you have to leave the files in `onnx_models` folder using the names: `detection_model.onnx` and `recognition_model.onnx` accordingly.

Model export assumes: 
- That only 1 batch will be worked with. Tests have not yet been performed in order to have models that support multiple barch.
- That the height and width dimensions of the images to be analyzed are dynamic. It is possible to optimize the models by exporting them with fixed dimensions. This requires modification of the code.

## Branches of this respositiory
- **master**: Unmodified version of EasyOCR v1.5, Unmodified version of EasyOCR 1.5.
- **easyocr_onnx**: Modifications of EasyOCR to support ONNX in various forms.

## Folders in easyocr_onnx branch
- **easyocr_hybrid_onnx**: Slightly modified version of EasyOCR to be able to export and use ONNX models. Uses PyTorch to preprocess and postprocess model input and output data. It is possible to deactivate ONNX and use EasyOCR as normal.
- **easyocr_only_onnx**: (WORK IN PROGRESS) Highly modified version of EasyOCR that completely eliminates the use of PyTorch.
- **onxx_models**: Place where exported models are saved when using "easyocr_hybrid_onnx". The detection and recognition model **must be** present here for both, "easyocr_hybrid_onnx" and "easyocr_only_onnx" to function correctly.

## Requirements
- Docker
- Docker Compose
- Nvidia card with CUDA 11 and their drivers. 

In case of not complying with the CUDA requirement, it is necessary to remove from line 11 to 17 of docker-compose.yml. In this case it will not be possible to export models to ONNX.

To ensure that you have CUDA correctly installed use the command: `nvidia-smi`. If a table is displayed where in the first line on the left it says "CUDA Version: 11.X", everything is correct. If not, check that you have installed the CUDA drivers correctly.

## Installation and Use

### Build and Run the Docker container
1. `docker-compose build`
2. `docker-compose up`

### Containers
Docker-compose raises two containers, one for each way to incorporate ONNX in EasyOCR:

- **easyocr_hybrid_onnx**: Slightly modified version of EasyOCR to be able to export and use ONNX models. Uses PyTorch to preprocess and postprocess model input and output data. It is possible to deactivate ONNX and use EasyOCR as normal.
- **easyocr_only_onnx**: (WORK IN PROGRESS) Highly modified version of EasyOCR that completely eliminates the use of PyTorch.
- **easypcr_compare_onnx**: Environment where easyocr_hybrid_onnx and easyocr_only_onnx can be used to compare results.

### Interacting with easyocr_hybrid_onnx

1. In a new terminal, enter to the container shell: `docker exec -ti easyocr_hybrid_onnx /bin/bash`

- Run `python3 easyocr_hybrid_onnx/onnx_export.py` to export models. The exported models will be stored in: `onnx_models` folder. Now the method *readtext* has the *onnx_export* (bool) parameter to enable or disable the export of the ONNX model. Exporting will fail if you do not have CUDA.

- Run `python3 easyocr_hybrid_onnx/easyocr_with_onnx_model.py` to use EasyOCR with ONNX models. Now the method *readtext* has the *onnx* (bool) parameter to enable or disable the use of ONNX models.

### Interacting with easyocr_only_onnx

(WORK IN PROGRESS)

### Interacting with easypcr_compare_onnx

1. In a new terminal, enter to the container shell: `docker exec -ti easyocr_compare_onnx /bin/bash`
2. Check that you have the correct language ONNX models in: `onnx_models` (`detection_model.onnx` and `recognition_model.onnx`). If you don't have the models yet, use `python3 easyocr_hybrid_onnx/onnx_export.py`.
3. Edit in `compare.py` the language to use and the image to be tested. (Put the image in the root directory of the project).
4. Run `python3 compare.py`

The result issued by both, standard EasyOCR (PyTorch) and EasyOCR using only ONNX, will be displayed.

## Additional comments

- [Poetry](https://python-poetry.org/) is being used for Python package installation.
- The system does not support exporting and/or using models for different language sets simultaneously.
- This repository is based on [this guide](https://github.com/JaidedAI/EasyOCR/issues/746) I created, explaining the process.

