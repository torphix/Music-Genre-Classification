import os
import shutil
import pathlib
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


def load_dataset(df_dir, batch_size):
    # Format data into keras folder structure 
    print('Loading Dataset...')
    X_train_df = pd.read_csv(f"{df_dir}/X_train.csv")
    X_val_df = pd.read_csv(f"{df_dir}/X_val.csv")
    X_test_df = pd.read_csv(f"{df_dir}/X_test.csv")
    # Copy files into tempdir
    root_dir = pathlib.Path(__file__).parent.parent
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
			batch_size=batch_size,
			subset='training',
			class_mode='categorical')
    val_dl = datagen.flow_from_directory(
			'data/keras_dataset/val',
			target_size=(256,256),
			batch_size=batch_size,
			subset='validation',
			class_mode='categorical')
    test_dl = datagen.flow_from_directory(
			'data/keras_dataset/test',
			target_size=(256,256),
			batch_size=batch_size,
			class_mode='categorical')
    return train_dl, val_dl, test_dl
