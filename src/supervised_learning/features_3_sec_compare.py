import pickle
import pathlib
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from src.preprocessing import Preprocessor
from src.supervised_learning.parameter_search import search_svm_hyperparams
from src.supervised_learning.supervised_learning import compute_metrics


def preprocess(path):
    '''
    Applies preprocessing to features_3_sec
    '''
    with open(f'{path}/label_encoder.pickle', 'rb') as f:
        enc = pickle.load(f)

    df_3 = Preprocessor(fname='features_3_sec.csv').scale_features()

    # adding fname of the original file so that it matches df_30['filename']
    df_3['fname'] = df_3['filename'].apply(lambda x: ''.join(x.rsplit('.', 2)[:1] + ['.wav']))
    df_3['label'] = enc.transform(df_3['label'])

    #print(df_3.groupby('label').count(), df_30.groupby('label').count())

    # for 10 tracks we only have 9 3_sec samples
    # print(df_3.groupby('fname')['fname'].count().sort_values(ascending=True))

    # encoding fnames
    # enc = LabelEncoder()
    #df_30['filename'] = enc.fit_transform(df_30['filename'])
    #df_3['fname'] = enc.transform(df_3['fname'])

    return df_3


def train_test_split(df_3):
    '''
    Splits 3-sec in train and test set (according to the pre-split features_30_sec)
    '''
    # loading train, test for 30-sec
    path = pathlib.Path(__file__).parent.parent.parent
    path = f'{path}/data/train_test_val_split'
    X_30_train = pd.read_csv(f'{path}/X_train.csv')
    X_30_test = pd.read_csv(f'{path}/X_test.csv')
    y_30_train = np.ravel(pd.read_csv(f'{path}/y_train.csv'))
    y_30_test = np.ravel(pd.read_csv(f'{path}/y_test.csv'))

    # grouping 3-sec by filename
    df_3_by_fname = df_3.groupby('fname')

    train_names = X_30_train['filename']
    test_names = X_30_test['filename']

    X_30_train = X_30_train
    X_30_test = X_30_test

    # splitting 3-sec in train and test set
    X_3_train = pd.concat([df_3_by_fname.get_group(name) for name in train_names]).drop(columns=['label'])
    X_3_test = pd.concat([df_3_by_fname.get_group(name) for name in np.array(test_names)]).drop(columns=['label'])
    y_3_train = pd.concat([df_3_by_fname.get_group(name) for name in train_names])['label']
    y_3_test = pd.concat([df_3_by_fname.get_group(name) for name in np.array(test_names)])['label']

    return X_3_train, X_3_test, y_3_train, y_3_test, X_30_train, X_30_test, y_30_train, y_30_test


def fit_models():
    '''
    Fits svm with custom parameters (see parameter_search) to predict labels from both 30-sec and 3-sec
    Then groups 3-sec by file and outputs a single label per file.
    '''
    root_path = pathlib.Path(__file__).parent.parent.parent
    path = f'{root_path}/data/train_test_val_split'

    # model_params = search_svm_hyperparams()[0]
    model_params = {'C': 2.782559402207126, 'kernel': 'rbf'}

    d3 = preprocess(path)

    X_3_train, X_3_test, y_3_train, y_3_test, X_30_train, X_30_test, y_30_train, y_30_test = train_test_split(d3)

    svm_3 = SVC(**model_params).fit(X_3_train.select_dtypes(include=np.number), y_3_train)
    svm_30 = SVC(**model_params).fit(X_30_train.select_dtypes(include=np.number), y_30_train)

    pred_30 = svm_30.predict(X_30_test.select_dtypes(include=np.number))
    pred_3 = svm_3.predict(X_3_test.select_dtypes(include=np.number))

    acc_3_before_aggregation = accuracy_score(y_3_test, pred_3)

    pred_3_by_track = pd.DataFrame(zip(X_3_test['fname'], pred_3), columns=['fname', 'pred']).groupby('fname', sort=False)
    pred_3 = pred_3_by_track.aggregate(func=lambda x: x.mode()[0])

    # cm_3 = confusion_matrix(y_30_test, pred_3, normalize='true')
    # cm_30 = confusion_matrix(y_30_test, pred_30, normalize='true')
    #
    # fig, ax = plt.subplots(2)
    # sns.heatmap(cm_3, ax=ax[0], annot=True, square=True)
    # sns.heatmap(cm_30, ax=ax[1], annot=True, square=True)
    # print(pred_3, X_30_test['filename'])

    metrics_3 = compute_metrics(y_30_test, pred_3, path)
    metrics_30 = compute_metrics(y_30_test, pred_30, path)

    # accuracy_by_class = pd.DataFrame(np.diag(cm_30) / np.sum(cm_30, axis=1), columns=)
    # print(accuracy_by_class)
    # plt.show()
    # print(X_30_test['filename'], pred_3['fname'])

    return acc_3_before_aggregation, metrics_3, metrics_30




