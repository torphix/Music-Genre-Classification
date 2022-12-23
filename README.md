## Setup
```
conda env create -f conda.yaml
conda activate music
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
pip install -r requirements.txt
In order to configure different parameters such as input datatypes, model sizes etc look under configs/keras_config.yaml

<!-- python -m venv venv -->
<!-- source venv/bin/activate -->
<!-- pip install -r requirements.txt -->
```

## Commands
- run command ```python main.py ui``` to launch the UI providing a succincte overview of the project.

- run command ```python main.py train_nn ``` looks into the configs directory and runs as many neural network training runs as there are config.yaml files

- run command ```python main.py fit_classical_models``` to run the classical machine learning models

- run command ```python main.py evaluate_classical_models``` to evaluate the classical machine learning models

- open notebook `Classical ML data analysis.ipynb` for a brief analysis with classical ml models.


## TODO:
- Add preprocessing pipeline to one command
