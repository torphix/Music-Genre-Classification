import os
import yaml
import librosa
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from .model import make_model_2d
from tensorflow.keras import layers
from src.preprocessing import Preprocessor
from src.neural_network_keras.data import load_dataset
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


class TFTrainer:
    def __init__(self):
        with open('configs/keras_config.yaml', 'r') as f:
            self.config = yaml.load(f.read(), Loader=yaml.FullLoader)

        if self.config['ckpt_path'] is not None and self.config['ckpt_path'] != '':
            print('Loading Model From CKPT')
            self.model = keras.models.load_model(self.config['ckpt_path'])
        else:
            self.model = make_model_2d(input_shape=(256,256,3), num_classes=10, resnet_type=self.config['resnet_type'])

        self.train_ds, self.val_ds, self.test_ds = load_dataset('data/train_test_val_split_short_files', self.config['batch_size'])

    def run(self):
        callbacks = [
            keras.callbacks.ModelCheckpoint(f"logs/keras/run-{len(os.listdir('logs/keras'))+1}" + "/epoch-{epoch}.keras"),
            keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5),
            keras.callbacks.ReduceLROnPlateau(
                                monitor='val_loss', 
                                factor=0.2,
                                patience=5, 
                                min_lr=0.001)
        ]
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config['learning_rate']),
            loss=keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=["acc"],
        )
        for x in self.val_ds:
            print(x)
        history = self.model.fit(
            self.train_ds,
            epochs=self.config['epochs'],
            callbacks=callbacks,
            validation_data=self.val_ds,
            validation_freq=4,
        )
        print('Evaluating on Validation')
        self.model.evaluate(self.val_ds)
        print('Evaluating on Testset')
        self.model.evaluate(self.test_ds)

        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


    def inference_wav_file(self, wav_path, ckpt_path):
        model = keras.models.load_model(ckpt_path)
        # Convert to correct data format
        wav, sr = librosa.load(wav_path)

        Preprocessor.mel_to_img()

        model.predict
        # TODO add wav file inference here


    def inference_dir(self, ckpt_path, data_dir):
        model = keras.models.load_model(ckpt_path)

        datagen = ImageDataGenerator(rescale=1./255, fill_mode='nearest')
        test_dl = datagen.flow_from_directory(
                data_dir,
                target_size=(256,256),
                batch_size=16,
                class_mode='categorical')
        model.evaluate(test_dl)
