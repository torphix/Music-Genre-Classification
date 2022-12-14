import os
import torch
import pathlib
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
import pickle


class Preprocessor:
    def __init__(self, data_path='data') -> None:
        root_dir = pathlib.Path(__file__).parent.parent
        self.data_path = f'{root_dir}/{data_path}'
        df = pd.read_csv(f'{self.data_path}/features_30_sec.csv')
        # Drop length as uncorrelated variable
        df = df.drop(labels=['length'], axis=1)
        df['label'] = df['filename'].apply(lambda fname: fname.split(".")[0])

        self.df = df

    @staticmethod
    def mel_to_img(mel):
        def _scale_minmax(X, min=0.0, max=1.0):
            X_std = (X - X.min()) / (X.max() - X.min())
            X_scaled = X_std * (max - min) + min
            return X_scaled
        # min-max scale to fit inside 8-bit range
        img = _scale_minmax(mel, 0, 255).astype(np.uint8)
        img = np.flip(img, axis=0) # put low frequencies at the bottom in image
        return 255-img

    def initial_preprocessing(self):
        pass

    def extract_mel_spectrogram(self, progress_bar_callback=None):  
        for i, genre in enumerate(tqdm(os.listdir(f'{self.data_path}/genres_original'), 'Extracting Mel Spectrograms')):
            if progress_bar_callback is not None:
                progress_bar_callback.progress(i*10)
            os.makedirs(f'{self.data_path}/mel_specs/{genre}', exist_ok=True)
            for file in os.listdir(f'{self.data_path}/genres_original/{genre}'):
                try:
                    wav, sr = librosa.load(f'{self.data_path}/genres_original/{genre}/{file}')
                    wav = wav / np.max(wav)
                except:
                    # Corrupted wav file
                    os.remove(f'{self.data_path}/genres_original/{genre}/{file}')
                mel_spec = librosa.feature.melspectrogram(y=wav, sr=sr)
                mel_spec = torch.tensor(mel_spec)
                # Normalise mel
                mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
                torch.save(mel_spec[:,:1293], f'{self.data_path}/mel_specs/{genre}/{file.split(".")[1]}.pt')


    def scale_features(self):
        # Substituting variance with standard deviation
        df = self.df
        var_features = [col for col in df.columns if col.endswith('var')]
        df[var_features] = np.sqrt(df[var_features])
        df.rename(columns={x: ''.join([x.rstrip('var'), 'stdev']) for x in var_features}, inplace=True)
        # Scaling features
        numerical_features = df.select_dtypes(np.number).columns
        df[numerical_features] = StandardScaler().fit_transform(df[numerical_features])
        # Saving
        # df.to_csv(f'{self.data_path}/features_30_sec_scaled.csv', index=False)
        return df.copy()

    def train_test_validation_split(self):
        df = self.scale_features()
        enc = LabelEncoder()
        audio_files = [file for genre in os.listdir(f'{self.data_path}/genres_original') 
                       for file in os.listdir(f'{self.data_path}/genres_original/{genre}')]
        df = df.sample(frac=1).reset_index(drop=True)
        X = df.loc[:, df.columns != 'label']
        X = df[df['filename'].isin(audio_files)]
        y = enc.fit_transform(X['label'])
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.9, random_state=42, stratify=X['label'])
        print(len(X_train), len(X_val), len(y_train), len(y_val))
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, train_size=0.5, random_state=101, stratify=X_val['label'])
        print('Label Proportions Train:', X_train['label'].value_counts())
        print('Label Proportions Val:', X_val['label'].value_counts())
        print('Label Proportions Test:', X_test['label'].value_counts())
        os.makedirs(f'{self.data_path}/train_test_val_split', exist_ok=True)
        for fname, f in {'X_train': X_train, 'X_val': X_val, 'X_test': X_test, 'y_train': y_train, 'y_val': y_val, 'y_test': y_test}.items():
            f = pd.DataFrame(f)
            f.to_csv(f'{self.data_path}/train_test_val_split/{fname}.csv', index=False)

        with open(f'{self.data_path}/train_test_val_split/label_encoder.pickle', 'wb') as f:
            pickle.dump(enc, f)
