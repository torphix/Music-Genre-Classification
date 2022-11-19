## General guidelines when pushing updates
0) Ensure to activate venv and download pip3 install -r requirements.txt
1) If installing new packages remeber to pip freeze > requirements.txt after

## Commands
- run command ```python main.py ui``` to launch the UI providing a succincte overview of the project all of the following commands can be interacted with through the UI or manually through the command line

- run command ```python main.py train -m='resnet'``` modify config.yaml to configure the model hyperparamters

## TODO
1) Check accuracy metrics are correct val loss is behaving strangely
2) Add more input features to neural network
3) Get Increase accuracy of unsupervised clustering and present visualisations


## In report
0) Some files where dropped due to corrputed audio
1) Why used adam optimizer + (brief descriptor)
2) Bugs faced:  
    - Passed a tiny momentum value to optimizer essentially reducing the inertia of the optimizer steps to nill
    - Was passing onehot class label encodings instead of the true indices to the loss function resulting in incorrect optimisation

