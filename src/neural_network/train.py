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
from torch.utils.data import DataLoader, random_split
from .data import ImageDataset, MelDataset, AudioDataset


class Trainer:
    def __init__(self):

        with open('config.yaml', 'r') as f:
            self.config = yaml.load(f.read(), Loader=yaml.FullLoader)['neural_network']
            
        assert self.config['data_type'] in ['img', 'mel', 'audio'], \
            f'data_type: {self.config["data_type"]} not in available options'

        torch.manual_seed(self.config['seed'])
        # Paramters
        self.epochs = self.config['epochs']
        self.log_step = self.config['log_step']

        logging.info('Recording training logs to directory: logs/')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load and split data
        dataset, in_d = self.load_dataset(self.config['data_type'])
        val_size = int(self.config['val_split_size'] * dataset.__len__())
        train_size = int(dataset.__len__() - val_size)
        logging.info(f'Splitting dataset: train_size: {train_size}, val_size: {val_size}')
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        self.train_dataloader = DataLoader(train_dataset, self.config['batch_size'], True, collate_fn=dataset.collate_fn)
        self.val_dataloader = DataLoader(val_dataset, self.config['batch_size'], True, collate_fn=dataset.collate_fn)

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
        return model

    def load_dataset(self, data_type):
        if data_type == 'mel':
            in_d = 128
            dataset = MelDataset('data')
        elif data_type == 'img':
            in_d = 4
            dataset = ImageDataset('data')
        elif data_type == 'audio':
            in_d = 1
            dataset = AudioDataset('data')
        return dataset, in_d

    def __call__(self, 
                train_callback=None, train_callback_args=(),
                val_callback=None, val_callback_args=()):
        self.val_loss, self.val_acc = self.val_iter()
        start_val_loss, start_val_acc = self.val_loss, self.val_acc
        val_losses = {0:self.val_loss.detach().cpu()}
        train_losses = {}
        for e in range(self.epochs):
            # Validation
            if e % self.log_step == 0:
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
        logging.info(f'''
            Training Complete. Start Val loss: {start_val_loss} Final Val loss: {final_val_loss} \
             Start Val Accuracy: {start_val_acc}  Final Val Accuracy: {final_val_acc} 
            ''')
        os.makedirs(f'logs/{e+1}', exist_ok=True)
        torch.save(self.model, f'logs/{e+1}/model.ckpt')
        torch.save({'train_losses': train_losses, 'val_losses': val_losses}, f'logs/{e+1}/losses.pt')
        with open(f'logs/{e+1}/config.json', 'w') as f:
            f.write(json.dumps({
                'model_parameters': self.count_parameters(),
                'architechture': self.config['architechture'],
                'epochs': e+1,
                'batch_size': self.config['batch_size'],
            }))

    def train_iter(self, epoch):
        train_progress_bar = tqdm(self.train_dataloader, 'Train Epoch')
        running_loss = 0
        self.model = self.model.train()
        for i, data in enumerate(train_progress_bar):
            inputs, targets = data['inputs'].to(self.device), data['targets'].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            running_loss += loss
            if i % self.log_step == 0:
                train_acc = self.calc_accuracy(outputs, targets)
                train_progress_bar.set_description(
                    f'Epoch: {epoch}/{self.epochs}, \
                      Train Loss: {round(loss.item(), 4)} \
                      Val Loss: {round(self.val_loss.item(), 4)}\
                      Val Acc: {round(self.val_acc.item(), 4)}'
                    )
        return running_loss / (self.train_dataloader.__len__()), train_acc

    def val_iter(self):
        logging.info('Running Validation..')
        self.model = self.model.eval()
        val_loss, val_acc = 0, 0
        for data in self.val_dataloader:
            with torch.no_grad():
                inputs, targets = data['inputs'].to(self.device), data['targets'].to(self.device)
                outputs = self.model(inputs)
                val_loss += self.criterion(outputs, targets)
                val_acc += self.calc_accuracy(outputs, targets)
        val_loss /= self.val_dataloader.__len__()
        val_acc /= len(self.val_dataloader)
        return val_loss, val_acc

    def calc_accuracy(self, outputs, targets):
        outputs = torch.argmax(outputs, dim=1)
        targets = torch.argmax(targets, dim=1)
        correct_vals = torch.sum(outputs == targets)
        total_vals = outputs.shape[0]
        return correct_vals / total_vals

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)