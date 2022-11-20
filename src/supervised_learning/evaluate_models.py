import pathlib
from src.supervised_learning.supervised_learning import *


def evaluate_models(path_to_data):

    metrics = {}
    for c in ['gini', 'entropy']:
        metrics[f'decision_tree_{c}'] = decision_tree_classifier(path_to_data, **{'criterion': c})

    metrics['logistic_regression'] = logistic_regression_classifier(path_to_data)
    metrics['svm'] = svm_classifier(path_to_data)

    for k in range(3, 15, 3):
        metrics[f'knn_{k}'] = knn_classifier(path_to_data, **{'n_neighbors': k})

    for c in ['gini', 'entropy']:
        metrics[f'random_forest_{c}'] = random_forest_classifier(path_to_data, **{'criterion': c})

    metrics['naive_bayes'] = naive_bayes_classifier(path_to_data)

    for key, val in metrics.items():
        acc = val['accuracy']
        prec = val['precision']
        print(f'{key} accuracy: {acc:.3f}, precision: {prec:.3f}')


path = pathlib.Path(__file__).parent.parent
evaluate_models('../../data/train_test_val_split')