## General guidelines when pushing updates
0) Ensure to activate venv and download pip3 install -r requirements.txt
1) If installing new packages remeber to pip freeze > requirements.txt after

## Commands
- run command ```python main.py ui``` to launch the UI providing a succincte overview of the project all of the following commands can be interacted with through the UI or manually through the command line

- run command ```python main.py train -m='resnet'``` modify config.yaml to configure the model hyperparamters

## TODO
1) Add accuracy and precision metrics
2) Add more input features to neural network
3) Get Increase accuracy of unsupervised clustering and present visualisations
