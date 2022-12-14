import os
import json
import yaml
import torch
import logging
import torch.nn as nn
from tqdm import tqdm
import lovely_tensors as lt
import torch.nn.functional as F
from .data import load_dataloaders
from prettytable import PrettyTable
from .resnet import ResNet1d, ResNet2d
from torch.utils.data import DataLoader
from .multi_modal_net import MultiModalNet
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer:
    def __init__(self, config_path):
        # lt.monkey_patch()
        with open(config_path, 'r') as f:
            self.config = yaml.load(f.read(), Loader=yaml.FullLoader)['neural_network']
            
        assert self.config['data_type'] in ['img', 'mel', 'audio', 'multi_modal'], \
            f'data_type: {self.config["data_type"]} not in available options'

        self.genre_dict = {
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

        torch.manual_seed(self.config['seed'])
        # Paramters
        self.epochs = self.config['epochs']
        self.log_step = self.config['log_step']
        self.val_check_n_epochs = self.config['val_check_n_epochs']

        logging.info('Recording training logs to directory: logs/')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load and split data
        logging.info('Loading Datasets..')
        self.train_dl, self.val_dl, self.test_dl, in_d = load_dataloaders(self.config['data_type'], 
                                                                          self.config['batch_size'],
                                                                          self.config['use_n_seconds'])
        logging.info('Loading Model..')
        self.model = self.load_model(self.config['data_type'], in_d, 
                                    self.config['architechture'], 
                                    self.config['load_trained_model']).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['optim']['learning_rate'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, **self.config['scheduler'])
        logging.info(f'Starting training for: {self.config["epochs"]} epochs')
        logging.info(f'Number of parameters: {round(self.count_parameters()/1_000_000, 2)} M, {round(self.get_model_size(self.model), 2)}: MiB')
        self.metrics_table = PrettyTable(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'LR'])
        self.metrics_table.add_row([0.0,0.0,0.0,0.0,0.0,self.get_lr()])
        print(self.metrics_table)

    def __call__(self):
        train_metrics = self.init_metrics_dict()
        val_metrics = self.init_metrics_dict()
        for e in range(self.epochs):
            # Validation
            val_loss, val_acc = self.eval_iter(self.val_dl)
            val_metrics['loss'].append(val_loss.cpu())
            val_metrics['acc'].append(val_acc.cpu())
            # Train 
            train_loss, train_acc = self.train_iter(e)
            train_metrics['loss'].append(train_loss.detach().cpu())
            train_metrics['acc'].append(train_acc.detach().cpu())
            # Reduce LR & Log
            self.scheduler.step(val_acc)
            self.log_metrics(e+1, train_loss, train_acc, val_loss, val_acc, self.get_lr())
        # Test
        final_test_loss, final_test_acc, cf_matrix = self.eval_iter(self.test_dl, True)
        logging.info(f'Testset Accuracy: {final_test_acc}')
        # Save training run
        folder_name = f'{e+1}-{self.config["data_type"]}-{self.config["architechture"]}'
        os.makedirs(f'logs/{folder_name}', exist_ok=True)
        torch.save(self.model.state_dict(), f'logs/{folder_name}/model.ckpt')
        torch.save(train_metrics, f'logs/{folder_name}/train_metrics.pt')
        torch.save(val_metrics, f'logs/{folder_name}/val_metrics.pt')
        with open(f'logs/{folder_name}/config.json', 'w') as f:
            f.write(json.dumps({
                'model_parameters': self.count_parameters(),
                'architechture': self.config['architechture'],
                'epochs': e+1,
                'batch_size': self.config['batch_size'],
                'test_acc':final_test_acc.detach().cpu().item(),
                'cf_matrix':cf_matrix.tolist(),
            }))   

    def train_iter(self, epoch):
        train_progress_bar = tqdm(self.train_dl, 
                                  f'Epoch: {epoch}/{self.epochs}', 
                                  leave=False)
        running_loss, running_acc = 0, 0
        for i, data in enumerate(train_progress_bar):
            inputs, targets = self.to_device(data)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss
            running_acc += self.calc_accuracy(outputs, targets)
        train_acc = running_acc / len(self.train_dl)
        train_loss = running_loss / (self.train_dl.__len__())
        return train_loss, train_acc

    def eval_iter(self, dataloader, plot_confusion=False):
        eval_loss, eval_acc = 0, 0
        outputs, targets = [],[]
        with torch.no_grad():
            for data in dataloader:
                x, y = self.to_device(data)
                y_pred = self.model(x)
                eval_loss += self.criterion(y_pred, y)
                eval_acc += self.calc_accuracy(y_pred, y)
                if plot_confusion:
                    outputs.append(torch.max(y_pred, dim=1)[1])
                    targets.append(y)
        
        eval_loss /= dataloader.__len__()
        eval_acc /= len(dataloader)
        if plot_confusion:
            outputs = torch.cat(outputs).flatten().detach().cpu().tolist().map(lambda x: self.genre_dict[x])
            targets = torch.cat(targets).flatten().detach().cpu().tolist().map(lambda x: self.genre_dict[x])
            cf_matrix = confusion_matrix(targets, outputs, labels=list(self.genre_dict.keys()))
            return eval_loss, eval_acc, cf_matrix
        else:
            return eval_loss, eval_acc

    def calc_accuracy(self, outputs, targets):
        # Accuracy
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

    def get_model_size(self, model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    def init_metrics_dict(self, metrics=['loss','acc']):
        return {metric: [] for metric in metrics}

    def log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        self.metrics_table.add_row([
            epoch,
            round(train_loss.detach().item(), 5), 
            f'{round(train_acc.detach().item(), 2)}%', 
            round(val_loss.detach().item(), 5), 
            f'{round(val_acc.detach().item(), 2)}%', 
            round(lr, 8)])
        print( "\n".join(self.metrics_table.get_string().splitlines()[-2:])) # Print only new row


    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def load_model(self, data_type, in_d, architechture, load_trained_model=''):
        if architechture == 'resnet18':
            n_layers = [2, 2, 2, 2]
        elif architechture == 'resnet50':
            n_layers = [3, 4, 6, 3]
        elif architechture == 'resnet101':
            n_layers = [3, 4, 23, 3]
        elif architechture == 'resnet152':
            n_layers = [3, 8, 36, 3]
        if data_type == 'audio' or data_type == 'mel':
            model = ResNet1d(in_d=in_d, n_layers=n_layers, n_classes=10)
        elif data_type == 'img':
            model = ResNet2d(in_d=in_d, n_layers=n_layers, n_classes=10)
        elif data_type == 'multi_modal':
            model = MultiModalNet(n_layers)
        if load_trained_model != '' and load_trained_model != None:
            model.load_state_dict(torch.load(load_trained_model))
        return model

