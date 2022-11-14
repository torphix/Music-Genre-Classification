import os
import torch
import librosa
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class MelDataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.data_path = f'{data_path}/mel_specs'
        self.files = [f'{folder}/{file}' 
                        for folder in os.listdir(self.data_path) 
                        for file in os.listdir(f'{self.data_path}/{folder}')]

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

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        fname =  self.files[index]
        genre, index = fname.split("/")
        mel = torch.load(f'{self.data_path}/{genre}/{index}') 
        target = [0 for _ in range(len(self.target_dict))]       
        target[self.target_dict[genre]] = 1
        return {
            'target': torch.tensor(target),
            'mel':mel,
        }

    def collate_fn(self, data):
        mels, targets = [], []
        # Crop mel lengths to be the same
        for d in data:
            d['mel'] = d['mel'][:,:1293] # Crop to 30s
            if d['mel'].shape[1] < 1293:
                mels.append(F.pad(d['mel'], (0, 1293-d['mel'].shape[1]), 'constant', 0))
            else:
                mels.append(d['mel'])
            targets.append(d['target'])
        return {
            'targets': torch.stack(targets).float(),
            'inputs': torch.stack(mels)
        }


class AudioDataset(Dataset):
    def __init__(self, data_path, use_n_seconds=10) -> None:
        super().__init__()
        self.data_path = f'{data_path}/genres_original'
        self.use_n_seconds = use_n_seconds
        self.files = [f'{folder}/{file}' 
                        for folder in os.listdir(self.data_path) 
                        for file in os.listdir(f'{self.data_path}/{folder}')]

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

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        fname =  self.files[index]
        genre, index = fname.split("/")
        wav, sr = librosa.load(f'{self.data_path}/{genre}/{index}') 
        wav /= torch.max(torch.tensor(wav))
        target = [0 for _ in range(len(self.target_dict))]       
        target[self.target_dict[genre]] = 1
        return {
            'target': torch.tensor(target),
            'audio':wav[:int(self.use_n_seconds*sr)].unsqueeze(0),
        }

    def collate_fn(self, data):
        wavs, targets = [], []
        # Crop mel lengths to be the same
        for d in data:
            wavs.append(d['audio'])
            targets.append(d['target'])
        return {
            'targets': torch.stack(targets).float(),
            'inputs': torch.stack(wavs).float()
        }


class ImageDataset(Dataset):
    def __init__(self, data_path, use_n_seconds=2) -> None:
        super().__init__()
        self.data_path = f'{data_path}/images_original'
        self.use_n_seconds = use_n_seconds
        self.files = [f'{folder}/{file}' 
                        for folder in os.listdir(self.data_path) 
                        for file in os.listdir(f'{self.data_path}/{folder}')]
        self.image_transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
        ])

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

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        fname =  self.files[index]
        genre, index = fname.split("/")
        image = self.image_transforms(Image.open(f'{self.data_path}/{genre}/{index}'))
        image /= 255
        target = [0 for _ in range(len(self.target_dict))]       
        target[self.target_dict[genre]] = 1
        return {
            'target': torch.tensor(target),
            'image':image,
        }

    def collate_fn(self, data):
        images, targets = [], []
        # Crop mel lengths to be the same
        for d in data:
            images.append(d['image'])
            targets.append(d['target'])
        return {
            'targets': torch.stack(targets).float(),
            'inputs': torch.stack(images).float()
        }