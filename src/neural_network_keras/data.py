import os
import shutil
import pathlib
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, root_path, csv_path, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):


        self.root_path = root_path
        df = pd.read_csv(csv_path)
        self.labels = df['label']
        self.fnames = df['filename']

        self.target_dict = {
            'blues':0,
            'classical':1,
            'country':2,
            'disco':3,
            'hiphop':4,
            'jazz':5,
            'metal':6,
            'pop':7,
            'reggae':8,
            'rock':9
        }

        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.fnames) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


def load_dataset(df_dir, batch_size):
    # Format data into keras folder structure 
    print('Loading Dataset...')
    X_train_df = pd.read_csv(f"{df_dir}/X_train.csv")
    X_val_df = pd.read_csv(f"{df_dir}/X_val.csv")
    X_test_df = pd.read_csv(f"{df_dir}/X_test.csv")
    # Copy files into tempdir
    shutil.rmtree('data/keras_dataset', ignore_errors=True)
    for idx, row in X_train_df.iterrows():
        genre = row['label']
        fname = ".".join(row["filename"].split(".")[:-1])
        os.makedirs(f'data/keras_dataset/train/{genre}/', exist_ok=True)
        shutil.copyfile(f'data/images_split/{genre}/{fname}.png', 
                        f'data/keras_dataset/train/{genre}/{fname}.png')
    for idx, row in X_val_df.iterrows():
        genre = row['label']
        fname = ".".join(row["filename"].split(".")[:-1])
        os.makedirs(f'data/keras_dataset/val/{genre}/', exist_ok=True)
        shutil.copyfile(f'data/images_split/{genre}/{fname}.png', 
                        f'data/keras_dataset/val/{genre}/{fname}.png')
    for idx, row in X_test_df.iterrows():
        genre = row['label']
        fname = ".".join(row["filename"].split(".")[:-1])
        os.makedirs(f'data/keras_dataset/test/{genre}/', exist_ok=True)
        shutil.copyfile(f'data/images_split/{genre}/{fname}.png', 
                        f'data/keras_dataset/test/{genre}/{fname}.png')
	
    datagen = ImageDataGenerator(rescale=1./255, fill_mode='nearest')
	# flow_from_directory gets label for an image from the sub-directory it is placed in
	# Generate Train data
    train_dl = datagen.flow_from_directory(
			'data/keras_dataset/train',
			target_size=(256,256),
            shuffle=True,
			batch_size=batch_size,
			subset='training',
			class_mode='categorical')
    val_dl = datagen.flow_from_directory(
			'data/keras_dataset/val',
			target_size=(256,256),
			batch_size=128,
            shuffle=False,
			class_mode='categorical')
    test_dl = datagen.flow_from_directory(
			'data/keras_dataset/test',
			target_size=(256,256),
			batch_size=batch_size,
            shuffle=False,
			class_mode='categorical')
    return train_dl, val_dl, test_dl
