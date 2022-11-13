import os
import torch
import zipfile
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

    def initial_preprocessing(self):
        pass

    def extract_mel_spectrogram(self):  
        for genre in tqdm(os.listdir(f'{self.data_path}/genres_original'), 'Extracting Mel Spectrograms'):
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

        X = df.select_dtypes(np.number)
        y = enc.fit_transform(df['label'])
        X_, X_test, y_, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_, y_, train_size=0.75, random_state=101)

        for fname, f in {'X_train': X_train, 'X_val': X_val, 'X_test': X_test, 'y_train': y_train, 'y_val': y_val, 'y_test': y_test}.items():
            f = pd.DataFrame(f)
            f.to_csv(f'{self.data_path}/train_test_val_split/{fname}.csv', index=False)

        with open(f'{self.data_path}/train_test_val_split/label_encoder.pickle', 'wb') as f:
            pickle.dump(enc, f)
