import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from src.preprocessing import Preprocessor
from src.supervised_learning.logistic_regression_parameter_search import search_logit_hyperparams


def preprocess():
    df_3 = Preprocessor(fname='features_3_sec.csv').scale_features()
    df_30 = Preprocessor(fname='features_30_sec.csv').scale_features()

    # adding fname of the original file so that it matches df_30['filename']
    df_3['fname'] = df_3['filename'].apply(lambda x: ''.join(x.rsplit('.', 2)[:1] + ['.wav']))

    #print(df_3.groupby('label').count(), df_30.groupby('label').count())

    # for 10 tracks we only have 9 3_sec samples
    # print(df_3.groupby('fname')['fname'].count().sort_values(ascending=True))

    # encoding fnames
    # enc = LabelEncoder()
    #df_30['filename'] = enc.fit_transform(df_30['filename'])
    #df_3['fname'] = enc.transform(df_3['fname'])

    return df_3, df_30


def train_test_split(df_3, df_30, train_size_30=0.8):
    df_3_by_fname = df_3.groupby('fname')

    train_names = default_rng().choice(df_30['filename'], int(train_size_30 * df_30.shape[0]), replace=False)
    test_names = [n for n in df_30['filename'] if n not in train_names]
    X_30_train = df_30[df_30['filename'].isin(train_names)].drop(columns=['label'])
    X_30_test = df_30[df_30['filename'].isin(test_names)].drop(columns=['label'])
    y_30_train = df_30[df_30['filename'].isin(train_names)]['label']
    y_30_test = df_30[df_30['filename'].isin(test_names)]['label']

    X_3_train = pd.concat([df_3_by_fname.get_group(name) for name in train_names]).drop(columns=['label'])
    X_3_test = pd.concat([df_3_by_fname.get_group(name) for name in test_names]).drop(columns=['label'])
    y_3_train = pd.concat([df_3_by_fname.get_group(name) for name in train_names])['label']
    y_3_test = pd.concat([df_3_by_fname.get_group(name) for name in test_names])['label']

    return X_3_train, X_3_test, y_3_train, y_3_test, X_30_train, X_30_test, y_30_train, y_30_test


def fit_models():
    # model_params = search_logit_hyperparams()[0]
    model_params = {'multi_class': 'ovr', 'penalty': 'l2', 'C': 0.3593813663804626, 'max_iter': 4000}

    d3, d30 = preprocess()

    X_3_train, X_3_test, y_3_train, y_3_test, X_30_train, X_30_test, y_30_train, y_30_test = train_test_split(d3, d30)

    logreg_3 = LogisticRegression(**model_params).fit(X_3_train.select_dtypes(include=np.number), y_3_train)
    logreg_30 = LogisticRegression(**model_params).fit(X_30_train.select_dtypes(include=np.number), y_30_train)

    pred_30 = logreg_30.predict(X_30_test.select_dtypes(include=np.number))
    pred_3 = logreg_3.predict(X_3_test.select_dtypes(include=np.number))
    acc_3_before_aggregation = accuracy_score(y_3_test, pred_3)
    print(f'acc before aggregation: {acc_3_before_aggregation:.3f}')

    pred_3_by_track = pd.DataFrame(zip(X_3_test['fname'], pred_3), columns=['fname', 'pred']).groupby('fname')
    pred_3 = pred_3_by_track.aggregate(func=lambda x: x.mode()[0])

    accuracy_3 = accuracy_score(y_30_test, pred_3)
    accuracy_30 = accuracy_score(y_30_test, pred_30)
    print(f'acc_3: {accuracy_3}, acc_30: {accuracy_30}')

    # cm_3 = confusion_matrix(y_30_test, pred_3)
    # cm_30 = confusion_matrix(y_30_test, pred_30)

    # fig, ax = plt.subplots(2)
    # ConfusionMatrixDisplay(cm_3).plot(ax=ax[0])
    # ConfusionMatrixDisplay(cm_30).plot(ax=ax[1])
    # plt.show()


fit_models()

