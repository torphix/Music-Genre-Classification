## Setup
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Commands
- run command ```python main.py ui``` to launch the UI providing a succincte overview of the project.

- run command ```python main.py train_nn ``` looks into the configs directory and runs as many neural network training runs as there are config.yaml files

- run command ```python main.py fit_classical_models``` to run the classical machine learning models

- run command ```python main.py evaluate_classical_models``` to evaluate the classical machine learning models
