import os
import json
import yaml
import torch
import shutil
import logging
import datetime
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from .scheduler import CustomScheduler
from .resnet import ResNet1d, ResNet2d
from .multi_modal_net import MultiModalNet
from torch.utils.data import DataLoader, random_split
from .data import ImageDataset, MelDataset, AudioDataset, MultiModalDataset

class Trainer:
    def __init__(self):

        with open('config.yaml', 'r') as f:
            self.config = yaml.load(f.read(), Loader=yaml.FullLoader)['neural_network']
            
        assert self.config['data_type'] in ['img', 'mel', 'audio', 'multi_modal'], \
            f'data_type: {self.config["data_type"]} not in available options'

        torch.manual_seed(self.config['seed'])
        # Paramters
        self.epochs = self.config['epochs']
        self.log_step = self.config['log_step']
        self.val_check_n_epochs = self.config['val_check_n_epochs']

        logging.info('Recording training logs to directory: logs/')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load and split data
        logging.info('Loading Datasets..')
        self.train_dl, self.val_dl, self.test_dl, in_d = self.load_dataloaders(self.config['data_type'])
        logging.info('Loading Model..')
        self.model = self.load_model(self.config['data_type'], in_d, self.config['architechture']).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['optim']['learning_rate'])
        self.scheduler = CustomScheduler(self.optimizer, **self.config['scheduler'])
        logging.info(f'Starting training for: {self.config["epochs"]}')
        logging.info(f'Number of parameters: {self.count_parameters()}')

    def load_model(self, data_type, in_d, arch):
        if data_type == 'audio' or data_type == 'mel':
            model = ResNet1d(in_d=in_d, out_d=10, n_blocks=self.config['n_blocks'])
        elif data_type == 'img':
            model = ResNet2d(in_d=in_d, out_d=10, n_blocks=self.config['n_blocks'])
        elif data_type == 'multi_modal':
            model = MultiModalNet()
        return model

    def load_dataloaders(self, data_type):
        # TODO add loading of val/test/train dataloaders here
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
            train_dataset = MultiModalDataset('data', 'data/train_test_val_split/X_train.csv', 3)
            val_dataset = MultiModalDataset('data', 'data/train_test_val_split/X_val.csv', 3)
            test_dataset = MultiModalDataset('data', 'data/train_test_val_split/X_test.csv', 3)
            train_dl = DataLoader(train_dataset, self.config['batch_size'], True, collate_fn=train_dataset.collate_fn)
            val_dl = DataLoader(val_dataset, self.config['batch_size'], False, collate_fn=train_dataset.collate_fn)
            test_dl = DataLoader(test_dataset, self.config['batch_size'], False, collate_fn=train_dataset.collate_fn)
            in_d = None
        return train_dl, val_dl, test_dl, in_d

    def __call__(self, 
                train_callback=None, train_callback_args=(),
                val_callback=None, val_callback_args=()):
        self.val_loss, self.val_acc = self.val_iter()
        start_val_loss, start_val_acc = self.val_loss, self.val_acc
        logging.info(f'Starting Validation Accuracy: {round(start_val_acc.item(),2)}%')
        val_losses = {0:self.val_loss.detach().cpu()}
        train_losses = {}
        for e in range(self.epochs):
            # Validation
            if e % self.val_check_n_epochs == 0:
                self.val_loss, self.val_acc = self.val_iter()
                val_losses[e] = self.val_loss
                if val_callback is not None:
                    val_callback(self.val_loss, *val_callback_args)
                    
            train_loss, train_acc = self.train_iter(e)
            train_losses[e] = train_loss.detach().cpu()
            if train_callback is not None:
                train_callback(train_loss, *train_callback_args)

        final_val_loss, final_val_acc = self.val_iter()
        val_losses[e] = final_val_loss.detach().cpu()
        final_test_loss, final_test_acc = self.test_iter()
        logging.info(f'''
            Training Complete. Start Val loss: {start_val_loss} Final Val loss: {final_val_loss} \
             Start Val Accuracy: {start_val_acc}  Final Val Accuracy: {final_val_acc} 
             Testset Accuracy: {final_test_acc}
            ''')
        os.makedirs(f'logs/{e+1}-{self.config["data_type"]}', exist_ok=True)
        torch.save(self.model, f'logs/{e+1}-{self.config["data_type"]}/model.ckpt')
        torch.save({'train_losses': train_losses, 'val_losses': val_losses}, f'logs/{e+1}-{self.config["data_type"]}/losses.pt')
        with open(f'logs/{e+1}-{self.config["data_type"]}/config.json', 'w') as f:
            f.write(json.dumps({
                'model_parameters': self.count_parameters(),
                'architechture': self.config['architechture'],
                'epochs': e+1,
                'batch_size': self.config['batch_size'],
            }))

    def train_iter(self, epoch):
        train_progress_bar = tqdm(self.train_dl, 'Train Epoch')
        running_loss, running_acc = 0, 0
        self.model = self.model.train()
        for i, data in enumerate(train_progress_bar):
            inputs, targets = self.to_device(data)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            running_loss += loss
            running_acc += self.calc_accuracy(outputs, targets)
            if i % self.log_step == 0:
                train_progress_bar.set_description(
                    f'Epoch: {epoch}/{self.epochs}, \
                      Train Loss: {round(running_loss.item(), 4)} \
                      Train Acc: {round(running_acc.item(), 2)}% \
                      Val Loss: {round(self.val_loss.item(), 4)}\
                      Val Acc: {round(self.val_acc.item(), 2)}%'
                    )
        return running_loss / (self.test_dl.__len__()), running_acc / len(self.test_dl)

    def val_iter(self):
        logging.info('Running Validation..')
        self.model = self.model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for data in self.val_dl:
                inputs, targets = self.to_device(data)
                outputs = self.model(inputs)
                val_loss += self.criterion(outputs, targets)
                val_acc += self.calc_accuracy(outputs, targets)
        val_loss /= self.val_dl.__len__()
        val_acc /= len(self.val_dl)
        return val_loss, val_acc

    def test_iter(self):
        logging.info('Running Validation..')
        self.model = self.model.eval()
        test_loss, test_acc = 0, 0
        with torch.no_grad():
            for data in self.test_dl:
                inputs, targets = self.to_device(data)
                outputs = self.model(inputs)
                test_loss += self.criterion(outputs, targets)
                test_acc += self.calc_accuracy(outputs, targets)
        test_loss /= self.test_dl.__len__()
        test_acc /= len(self.test_dl)
        return test_loss, test_acc

    def calc_accuracy(self, outputs, targets):
        _, outputs = torch.max(outputs, dim=1)
        correct_vals = torch.sum(outputs == targets)
        total_vals = outputs.shape[0]
        return (correct_vals / total_vals) * 100

    def to_device(self, data):
        if isinstance(data['inputs'], dict):
            inputs = {k:v.to(self.device) for k,v in data['inputs'].items()}
            targets = data['targets'].to(self.device)
        else:
            inputs, targets = data['inputs'].to(self.device), data['targets'].to(self.device)
        return inputs, targets

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)