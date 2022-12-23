## Setup

- Note: code has been tested on linux and macOS we have tried to keep the environment as OS agnostic as possible however as we don't own a windows machine we cannot on windowsOS.
- Due to package conflicts between tensorflow and streamlit (package used to create UI) it is necessary to use a seperate environment for  viewing the UI
### Keras Neural Network Environment Setup
```
conda env create -f conda.yaml
conda activate music
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
pip install -r requirements/requirements_keras.txt
In order to configure different parameters such as input datatypes, model sizes etc look under configs/keras_config.yaml
```
### UI Environment Setup
Only required for running the UI and legacy torch code
```
conda deactivate
python -m venv venv
source venv/bin/activate
pip install -r requirements/requirements_ui.txt
```

## Commands

- run command ```python main.py ui``` (Activate UI environment first) to launch the UI providing a succinct overview of the project.

- run command ```python main.py process_data``` (Activate Keras Neural Network setup first)

- run command ```python main.py train_nn_keras ``` (Activate Keras Neural Network setup first) must run ```python main.py process_data``` first

- run command ```python main.py fit_classical_models``` to run the classical machine learning models (Activate Keras Neural Network setup first)

- run command ```python main.py evaluate_classical_models``` to evaluate the classical machine learning models (Activate Keras Neural Network setup first)

- open notebook `Classical ML data analysis.ipynb` for a brief analysis with classical ml models. 


