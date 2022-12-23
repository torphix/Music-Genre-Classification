import os
import yaml
import librosa
from tensorflow import keras
import matplotlib.pyplot as plt
from src.preprocessing import Preprocessor
from .model import load_model
from src.neural_network_keras.data import load_dataset
from keras.preprocessing.image import ImageDataGenerator


class TFTrainer:
    def __init__(self):
        with open('configs/keras_config.yaml', 'r') as f:
            self.config = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.model = load_model(self.config['data_type'], 
                                self.config['resnet_type'], 
                                self.config['ckpt_path'], 
                                self.config['use_pretrained_model'])

        self.train_ds, self.val_ds, self.test_ds = load_dataset(
                                                        'data/train_test_val_split_short_files', 
                                                        self.config['batch_size'], 
                                                        self.config['data_type'])

    def run(self):
        os.makedirs(f"logs/keras/", exist_ok=True)
        os.makedirs(f"logs/keras/run-{len(os.listdir('logs/keras'))+1}", exist_ok=True)
        callbacks = [
            keras.callbacks.ModelCheckpoint(f"logs/keras/run-{len(os.listdir('logs/keras'))+1}" + "/epoch-{epoch}.keras"),
            keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5),
            keras.callbacks.ReduceLROnPlateau(
                                monitor='val_loss', 
                                factor=0.2,
                                patience=3, 
                                min_lr=0.00001)
        ]
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config['learning_rate']),
            loss=keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=["acc"],
        )
        history = self.model.fit(
            self.train_ds,
            epochs=self.config['epochs'],
            callbacks=callbacks,
            validation_data=self.val_ds,
            validation_freq=self.config['check_val_n_epochs'],
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

    def inference_dir(self, ckpt_path, data_dir):
        model = keras.models.load_model(ckpt_path)
        datagen = ImageDataGenerator(rescale=1./255, fill_mode='nearest')
        test_dl = datagen.flow_from_directory(
                data_dir,
                target_size=(256,256),
                batch_size=16,
                class_mode='categorical')
        model.evaluate(test_dl)
