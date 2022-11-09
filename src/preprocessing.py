import os
import torch
import zipfile
import pathlib
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


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
        df.to_csv(f'{self.data_path}/features_30_sec_scaled.csv', index=False)
