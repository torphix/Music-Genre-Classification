import os
import torch
import logging
import numpy as np
from tqdm import tqdm
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


def cluster_kmeans(data_path, n_datapoints, crop_len):
    '''
    data_path: directory with subdirectories for each class
    n_datapoints: number of n_datapoints of each class to load into memory        
    crop_len: amount by which to crop each datapoint (useful to limit memory useage)
    '''
    n_classes = len(os.listdir(data_path))
    kmeans = KMeans(n_clusters=n_classes, random_state=42, max_iter=500)
    logging.info('Loading data...')
    label_dict = {
        'hiphop':0,
        'reggae':1,
        'blues':2,
        'jazz':3,
        'disco':4,
        'pop':5,
        'rock':6,
        'metal':7,
        'country':8,
        'classical':9,
    }
    datapoints, ground_truth_labels = [], []
    for subdir in tqdm(os.listdir(data_path)):
        for file in os.listdir(f'{data_path}/{subdir}')[:n_datapoints]:
            datapoints.append(torch.load(f'{data_path}/{subdir}/{file}')[:,:crop_len].reshape(-1).numpy())
            ground_truth_labels.append(label_dict[subdir])
    logging.info('Fitting model')
    predicted_labels = kmeans.fit_predict(datapoints)
    print(accuracy_score(ground_truth_labels, predicted_labels))