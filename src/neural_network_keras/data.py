import os
import shutil
import pathlib
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import keras
from src.preprocessing import Preprocessor


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, 
                root_path, 
                csv_path, 
                batch_size=32,
                n_classes=10, 
                shuffle=True):

        self.root_path = root_path
        df = pd.read_csv(csv_path)
        self.fnames = df['filename']

        self.classes_dict = {
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

        self.batch_size = batch_size
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

        fnames = [self.fnames[i] for i in indexes]
        # Generate data
        X, y = self.__data_generation(fnames)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.fnames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, fnames):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 128, 130))
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, fname in enumerate(fnames):
            mel = np.load(f'{self.root_path}/{fname.split(".")[1]}/{fname}')
            mel = np.concatenate((mel, np.zeros((128, 130-mel.shape[1]))), axis=1)
            X[i,] = mel
            y[i] = self.classes_dict[fname.split(".")[1]]
        return X, keras.utils.to_categorical(y, num_classes=len(self.classes_dict.keys()))


def load_dataset(df_dir, batch_size, data_type):

    # Format data into keras folder structure 
    print('Loading Dataset...')
    X_train_df = pd.read_csv(f"{df_dir}/X_train.csv")
    X_val_df = pd.read_csv(f"{df_dir}/X_val.csv")
    X_test_df = pd.read_csv(f"{df_dir}/X_test.csv")

    # Copy files into tempdir
    shutil.rmtree('data/keras_dataset', ignore_errors=True)
    for idx, row in X_train_df.iterrows():
        genre = row['label']
        fname = ".".join(row["filename"].split(".")[:-1]) + '.npy'
        os.makedirs(f'data/keras_dataset/train/{genre}/', exist_ok=True)
        shutil.copyfile(f'data/mel_specs/{genre}/{fname}', 
                        f'data/keras_dataset/train/{genre}/{fname}')
    for idx, row in X_val_df.iterrows():
        genre = row['label']
        fname = ".".join(row["filename"].split(".")[:-1]) + '.npy'
        os.makedirs(f'data/keras_dataset/val/{genre}/', exist_ok=True)
        shutil.copyfile(f'data/mel_specs/{genre}/{fname}', 
                        f'data/keras_dataset/val/{genre}/{fname}')
    for idx, row in X_test_df.iterrows():
        genre = row['label']
        fname = ".".join(row["filename"].split(".")[:-1]) + '.npy'
        os.makedirs(f'data/keras_dataset/test/{genre}/', exist_ok=True)
        shutil.copyfile(f'data/mel_specs/{genre}/{fname}', 
                        f'data/keras_dataset/test/{genre}/{fname}')

	# Generate Train data
    if data_type == 'img':
        Preprocessor.convert_mel_folder_to_img_static('data/keras_dataset/train/', 'data/keras_dataset/train/')
        Preprocessor.convert_mel_folder_to_img_static('data/keras_dataset/val/', 'data/keras_dataset/val/')
        Preprocessor.convert_mel_folder_to_img_static('data/keras_dataset/test/', 'data/keras_dataset/test/')
        datagen = ImageDataGenerator()
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
    elif data_type == 'mel':
        train_dl = DataGenerator('data/keras_dataset/train', 
                                'data/train_test_val_split_short_files/X_train.csv',
                                64, 
                                shuffle=True)
        val_dl = DataGenerator('data/keras_dataset/val', 
                                'data/train_test_val_split_short_files/X_val.csv',
                                64, 
                                shuffle=False)
        test_dl = DataGenerator('data/keras_dataset/test', 
                                'data/train_test_val_split_short_files/X_test.csv',
                                64, 
                                shuffle=False)
    return train_dl, val_dl, test_dl
