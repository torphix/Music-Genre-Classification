import os
import shutil
import pickle
import librosa
import pathlib
import skimage.io
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class Preprocessor:
    def __init__(self, data_path='data', fname='features_30_sec.csv') -> None:
        root_dir = pathlib.Path(__file__).parent.parent
        self.data_path = f'{root_dir}/{data_path}'
        df = pd.read_csv(f'{self.data_path}/{fname}')
        # Drop length as uncorrelated variable
        df = df.drop(labels=['length'], axis=1)
        df['label'] = df['filename'].apply(lambda fname: fname.split(".")[0])

        self.df = df

    def convert_mel_folder_to_img(self, in_dir, out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
        for genre in tqdm(os.listdir(in_dir), 'Processing'):
            os.makedirs(f'{out_dir}/{genre}', exist_ok=True)
            for file in os.listdir(f'{in_dir}/{genre}'):
                mel = Preprocessor.mel_to_img(np.load(f'{in_dir}/{genre}/{file}'))
                skimage.io.imsave(f'{out_dir}/{genre}/{".".join(file.split(".")[:-1])}.png', mel)
                
    @staticmethod
    def mel_to_img(mel):
        def _scale_minmax(X, min=0.0, max=1.0):
            X_std = (X - X.min()) / (X.max() - X.min())
            X_scaled = X_std * (max - min) + min
            return X_scaled
        # min-max scale to fit inside 8-bit range
        img = _scale_minmax(mel, 0, 255).astype(np.uint8)
        img = np.flip(img, axis=0) # put low frequencies at the bottom in image
        return np.expand_dims((255-img), axis=-1)

    def split_audio(self):
        '''
        Split audio files into 3 second clips to increase training data quantity
        '''
        shutil.rmtree(f'{self.data_path}/genres_split/', ignore_errors=True)
        for i, genre in enumerate(tqdm(os.listdir(f'{self.data_path}/genres_original'), 'Splitting audio')):
            os.makedirs(f'{self.data_path}/genres_split/{genre}', exist_ok=True)
            for file in os.listdir(f'{self.data_path}/genres_original/{genre}'):
                wav, sr = librosa.load(f'{self.data_path}/genres_original/{genre}/{file}')
                for idx,i in enumerate(range(0,30,3)):
                    wavfile.write(f'{self.data_path}/genres_split/{genre}/{idx}.{file}', sr, wav[i*sr:(i+3)*sr])

    def extract_mel_spectrogram(self, progress_bar_callback=None):  
        for i, genre in enumerate(tqdm(os.listdir(f'{self.data_path}/genres_split'), 'Extracting Mel Spectrograms')):
            if progress_bar_callback is not None:
                progress_bar_callback.progress(i*10)
            os.makedirs(f'{self.data_path}/mel_specs/{genre}', exist_ok=True)
            for file in os.listdir(f'{self.data_path}/genres_split/{genre}'):
                try:
                    wav, sr = librosa.load(f'{self.data_path}/genres_split/{genre}/{file}')
                    wav = wav / np.max(wav)
                except:
                    # Corrupted wav file
                    os.remove(f'{self.data_path}/genres_split/{genre}/{file}')
                mel_spec = librosa.feature.melspectrogram(y=wav, sr=sr)
                mel_spec = np.array(mel_spec)
                # Normalise mel
                mel_spec = np.log(np.clip(mel_spec, a_min=1e-5, a_max=np.max(mel_spec)))
                np.save(f'{self.data_path}/mel_specs/{genre}/{".".join(file.split(".")[0:-1])}.npy', mel_spec[:,:130])

    def create_genre_df(self):
        output_files, genre = [], []
        for folder in os.listdir('data/mel_specs'):
            for file in os.listdir(f'data/mel_specs/{folder}'):
                output_files.append(file)
                genre.append(folder)
        df = pd.DataFrame({
            'filename':output_files,
            'label':genre
        })
        df.to_csv(f'data/split_audio_file.csv')

    def scale_features(self):
        # Substituting variance with standard deviation
        df = self.df
        var_features = [col for col in df.columns if col.endswith('var')]
        df[var_features] = np.sqrt(df[var_features])
        df.rename(columns={x: ''.join([x.rstrip('var'), 'stdev']) for x in var_features}, inplace=True)
        # Scaling features
        numerical_features = df.select_dtypes(np.number).columns
        df[numerical_features] = StandardScaler().fit_transform(df[numerical_features])
        # Clamping
        df[numerical_features] = df[numerical_features].clip(lower=-5, upper=+5)
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
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42, stratify=X['label'])
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
