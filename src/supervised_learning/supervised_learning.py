import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture


def compute_metrics(y_test, y_pred, path_to_folder):
    """
    Computes accuracy, precision, confusion matrix.
    path_to_folder: path to folder containing data
    """
    with open(f'{path_to_folder}/label_encoder.pickle', 'rb') as f:
        enc = pickle.load(f)
    labels = enc.classes_
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred, normalize='true'), columns=labels, index=labels)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'confusion_matrix': cm}


def load_train_val(path_to_folder):
    """
    Loads train and validation set
    path_to_folder: path to folder containing data
    """
    X_train = pd.read_csv(f'{path_to_folder}/X_train.csv').select_dtypes(np.number)
    X_val = pd.read_csv(f'{path_to_folder}/X_val.csv').select_dtypes(np.number)
    y_train = np.ravel(pd.read_csv(f'{path_to_folder}/y_train.csv'))
    y_val = np.ravel(pd.read_csv(f'{path_to_folder}/y_val.csv'))

    return X_train, X_val, y_train, y_val


def decision_tree_classifier(path_to_data, **model_args):
    X_train, X_val, y_train, y_val = load_train_val(path_to_data)
    classifier = DecisionTreeClassifier(**model_args).fit(X_train, y_train)
    pred_labels = classifier.predict(X_val)

    return compute_metrics(y_val, pred_labels, path_to_data)


def logistic_regression_classifier(path_to_data, **model_args):
    X_train, X_val, y_train, y_val = load_train_val(path_to_data)
    classifier = LogisticRegression(multi_class='ovr', **model_args).fit(X_train, y_train)
    pred_labels = classifier.predict(X_val)

    return compute_metrics(y_val, pred_labels, path_to_data)


def svm_classifier(path_to_data, **model_args):
    X_train, X_val, y_train, y_val = load_train_val(path_to_data)
    classifier = SVC(**model_args).fit(X_train, y_train)
    pred_labels = classifier.predict(X_val)

    return compute_metrics(y_val, pred_labels, path_to_data)


def knn_classifier(path_to_data, **model_args):
    X_train, X_val, y_train, y_val = load_train_val(path_to_data)
    classifier = KNeighborsClassifier(**model_args).fit(X_train, y_train)
    pred_labels = classifier.predict(X_val)

    return compute_metrics(y_val, pred_labels, path_to_data)


def random_forest_classifier(path_to_data, **model_args):
    X_train, X_val, y_train, y_val = load_train_val(path_to_data)
    classifier = RandomForestClassifier(**model_args).fit(X_train, y_train)
    pred_labels = classifier.predict(X_val)

    return compute_metrics(y_val, pred_labels, path_to_data)


def naive_bayes_classifier(path_to_data, **model_args):
    X_train, X_val, y_train, y_val = load_train_val(path_to_data)
    classifier = GaussianNB(**model_args).fit(X_train, y_train)
    pred_labels = classifier.predict(X_val)

    return compute_metrics(y_val, pred_labels, path_to_data)


def gaussian_mixture_classifier(path_to_data, **model_args):
    X_train, X_val, y_train, y_val = load_train_val(path_to_data)
    classifier = GaussianMixture(n_components=10, max_iter=300, **model_args).fit(X_train, y_train)
    pred_labels = classifier.predict(X_val)

    return compute_metrics(y_val, pred_labels, path_to_data)
