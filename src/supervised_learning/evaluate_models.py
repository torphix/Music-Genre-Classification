import pathlib
from src.supervised_learning.supervised_learning import *


def evaluate_models(path_to_data):

    params = {'criterion': 'gini'}
    metrics = decision_tree_classifier(path_to_data, **params)

    for key, val in metrics.items():
        print(f'{key}: {val}')


path = pathlib.Path(__file__).parent.parent
evaluate_models('../../data/train_test_val_split')