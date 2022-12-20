import os
import torch
import librosa
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class MelDataset(Dataset):
    def __init__(self, root_path, csv_path) -> None:
        super().__init__()
        self.root_path = root_path
        df = pd.read_csv(csv_path)
        self.labels = df['label']
        self.fnames = df['filename']

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
        return len(self.fnames)
    
    def __getitem__(self, index):
        fname = self.fnames[index]
        label = self.labels[index]
        mel = torch.load(f'{self.root_path}/mel_specs/{label}/{fname}') 
        target = self.target_dict[label]
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
            'targets': torch.stack(targets),
            'inputs': torch.stack(mels)
        }


class AudioDataset(Dataset):
    def __init__(self, root_path, csv_path, use_n_seconds) -> None:
        super().__init__()
        self.use_n_seconds = use_n_seconds
        self.root_path = root_path
        self.df = pd.read_csv(csv_path)['filename']

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
        return len(self.df)
    
    def __getitem__(self, index):
        fname =  self.df[index]
        genre, index, _ = fname.split(".")
        wav, sr = librosa.load(f'{self.root_path}/genres_original/{genre}/{genre}.{index}.wav', sr=8000) 
        wav /= torch.max(torch.tensor(wav))
        target = self.target_dict[genre]
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
            'targets': torch.stack(targets),
            'inputs': torch.stack(wavs).float()
        }


class ImageDataset(Dataset):
    def __init__(self, root_path, csv_path) -> None:
        super().__init__()
        self.root_path = root_path
        self.df = pd.read_csv(csv_path)['filename']
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
        return len(self.df)
    
    def __getitem__(self, index):
        fname =  self.df[index]
        genre, index, _ = fname.split(".")
        image = self.image_transforms(Image.open(f'{self.root_path}/images_original/{genre}/{genre}{index}.png'))
        image /= 255
        target = self.target_dict[genre]
        return {
            'target': torch.tensor(target),
            'image':image[:3,:,:],
        }

    def collate_fn(self, data):
        images, targets = [], []
        # Crop mel lengths to be the same
        for d in data:
            images.append(d['image'])
            targets.append(d['target'])
        # return torch.stack(targets), torch.stack(images).float()
        return {
            'targets': torch.stack(targets),
            'inputs': torch.stack(images).float()
        }



class MultiModalDataset(Dataset):
    def __init__(self, root_path, csv_path, use_n_seconds=6) -> None:
        super().__init__()
        self.root_path = root_path
        self.df = pd.read_csv(csv_path)['filename']
        self.use_n_seconds = use_n_seconds
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
        return len(self.df)
    
    def __getitem__(self, index):
        fname =  self.df[index]
        genre, index, _ = fname.split(".")
        # Image
        image = self.image_transforms(Image.open(f'{self.root_path}/images_original/{genre}/{genre}{index}.png'))
        image /= 255
        # Audio
        wav, sr = librosa.load(f'{self.root_path}/genres_original/{genre}/{genre}.{index}.wav')
        wav /= torch.max(torch.tensor(wav))
        # Mel
        mel = torch.load(f'{self.root_path}/mel_specs/{genre}/{index}.pt') 
        mel /= 8
        target = self.target_dict[genre]
        return {
            'target': torch.tensor(target),
            'image':image,
            'mel':mel,
            'audio':wav[:int(self.use_n_seconds*sr)].unsqueeze(0),
        }

    def collate_fn(self, data):
        images, mels, audio, targets = [], [], [], []
        # Crop mel lengths to be the same
        for d in data:
            images.append(d['image'])
            targets.append(d['target'])
            # audio.append(d['audio'])
            # Mel
            d['mel'] = d['mel'][:,:1293] # Crop to 30s
            if d['mel'].shape[1] < 1293:
                mels.append(F.pad(d['mel'], (0, 1293-d['mel'].shape[1]), 'constant', 0))
            else:
                mels.append(d['mel'])
        return {
            'targets': torch.stack(targets),
            'inputs': {
                'images':torch.stack(images).float(),
                # 'audio':torch.stack(audio).float(),
                'mels':torch.stack(mels),
                }
        }


def load_dataloaders(data_type, batch_size, n_audio_seconds=6):
    if data_type == 'mel':
        in_d = 128
        train_dataset = MelDataset('data', 'data/train_test_val_split/X_train.csv')
        val_dataset = MelDataset('data', 'data/train_test_val_split/X_val.csv')
        test_dataset = MelDataset('data', 'data/train_test_val_split/X_test.csv')
    elif data_type == 'img':
        in_d = 4
        train_dataset = ImageDataset('data', 'data/train_test_val_split/X_train.csv')
        val_dataset = ImageDataset('data', 'data/train_test_val_split/X_val.csv')
        test_dataset = ImageDataset('data', 'data/train_test_val_split/X_test.csv')
    elif data_type == 'audio':
        in_d = 1
        train_dataset = AudioDataset('data', 'data/train_test_val_split/X_train.csv', 3)
        val_dataset = AudioDataset('data', 'data/train_test_val_split/X_val.csv', 3)
        test_dataset = AudioDataset('data', 'data/train_test_val_split/X_test.csv', 3)
    elif data_type == 'multi_modal':
        train_dataset = MultiModalDataset('data', 'data/train_test_val_split/X_train.csv', n_audio_seconds)
        val_dataset = MultiModalDataset('data', 'data/train_test_val_split/X_val.csv', n_audio_seconds)
        test_dataset = MultiModalDataset('data', 'data/train_test_val_split/X_test.csv', n_audio_seconds)
        in_d = None
    train_dl = DataLoader(train_dataset, batch_size, True, collate_fn=train_dataset.collate_fn)
    val_dl = DataLoader(val_dataset, batch_size, False, collate_fn=train_dataset.collate_fn)
    test_dl = DataLoader(test_dataset, batch_size, False, collate_fn=train_dataset.collate_fn)
    return train_dl, val_dl, test_dl, in_d