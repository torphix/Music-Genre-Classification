import zipfile
import pandas as pd


class Preprocessor:
    def __init__(self, data_path='./data') -> None:
        df = pd.read_csv(f'{data_path}/features_30_sec.csv')
        # Drop length as uncorrelated variable
        df = df.drop(labels=['length'], axis=1)
        df['label'] = df['filename'].apply(lambda fname: fname.split(".")[0])

        self.df = df
        print(self.df)

    def initial_preprocessing(self):
        pass
