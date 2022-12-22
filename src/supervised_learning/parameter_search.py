import pathlib
import numpy as np
from sklearn.svm import SVC
from supervised_learning import load_train_val
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression


def cross_validate_model(root_path, model, params=None):
    '''
    5-fold cross-validation on given model/hyperparams
    returns accuracy mean
    '''
    path = f'{root_path}/data/train_test_val_split'
    X_train, X_val, y_train, y_val = load_train_val(path)
    scores = cross_validate(model, X_train, y_train, scoring='accuracy', cv=5, fit_params=params)
    return np.mean(scores['test_score'])


def search_hyperparams(model, model_params):
    '''
    Finds the best set of hyperparams for the given model
    returns the best set of params and accuracy
    '''
    root_path = pathlib.Path(__file__).parent.parent.parent
    best_acc = 0
    best_model = None
    for params_set in model_params:
        classifier = model(**params_set)
        acc = cross_validate_model(root_path, classifier)
        if acc > best_acc:
            best_model = params_set
            best_acc = acc
    return best_model, best_acc


def search_logit_hyperparams():
    '''
    Tuning logistic regression hyperparams
    '''
    penalties = ['l2', 'elasticnet', 'l1', 'none']
    c_values = np.logspace(start=-4, stop=4, num=10)
    model_params = []
    for p in penalties:
        for c in c_values:
            if p == 'elasticnet':
                model_params.append({'solver':'saga', 'l1_ratio':0.5, 'penalty':p, 'C':c, 'max_iter':8000})
            elif p == 'l1':
                model_params.append({'solver':'saga', 'penalty':p, 'C':c, 'max_iter':8000})
            elif p == 'none':
                model_params.append({'penalty':p, 'max_iter':8000})
            else:
                model_params.append({'penalty':p, 'C':c, 'max_iter':8000})
    mod, acc = search_hyperparams(LogisticRegression, model_params)
    print(f'best accuracy: {acc}, best model hyperparams: {mod}')
    return mod, acc


def search_svm_hyperparams():
    '''
    Tuning support vector machine hyperparams
    '''
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    c_values = np.logspace(start=-4, stop=4, num=10)
    model_params = []

    for k in kernels:
        for c in c_values:
            model_params.append({'C':c, 'kernel':k})

    mod, acc = search_hyperparams(SVC, model_params)
    print(f'best accuracy: {acc}, best model hyperparams: {mod}')
    return mod, acc

