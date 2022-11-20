import pathlib
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from supervised_learning import load_train_val


def cross_validate_model(root_path, model, params=None):
    path = f'{root_path}/data/train_test_val_split'
    X_train, X_val, y_train, y_val = load_train_val(path)
    scores = cross_validate(model, X_train, y_train, scoring='accuracy', cv=5, fit_params=params)
    return np.mean(scores['test_score'])


def search_hyperparams(model, model_params):
    root_path = pathlib.Path(__file__).parent.parent.parent
    best_acc = 0
    best_model = None
    for params_set in model_params:
        classifier = model(**params_set)
        acc = cross_validate_model(root_path, classifier)
        p = params_set['penalty']
        print(f'{p},{acc:.3f}')
        if acc > best_acc:
            best_model = params_set
            best_acc = acc
    return best_model, best_acc


def search_logit_hyperparams():
    penalties = ['l2', 'elasticnet', 'l1', 'none']
    c_values = np.logspace(start=-4, stop=4, num=10)
    model_params = []
    for p in penalties:
        for c in c_values:
            if p == 'elasticnet':
                model_params.append({'multi_class':'ovr', 'solver':'saga', 'l1_ratio':0.5, 'penalty':p, 'C':c, 'max_iter':8000})
            elif p == 'l1':
                model_params.append({'multi_class':'ovr', 'solver':'saga', 'penalty':p, 'C':c, 'max_iter':8000})
            else:
                model_params.append({'multi_class':'ovr', 'penalty':p, 'C':c, 'max_iter':4000})
    mod, acc = search_hyperparams(LogisticRegression, model_params)
    print(f'best accuracy: {acc}, best model hyperparams: {mod}')
    return mod, acc


search_logit_hyperparams()
