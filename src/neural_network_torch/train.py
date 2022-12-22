import os
import json
import yaml
import torch
import logging
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from .data import load_dataloaders
from prettytable import PrettyTable
from .resnet import ResNet1d, ResNet2d
from torch.utils.data import DataLoader
from .multi_modal_net import MultiModalNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet50 as pretrained_resnet50
from sklearn.metrics import confusion_matrix, precision_score, recall_score


class Trainer:
    def __init__(self, config_path):
        # lt.monkey_patch()
        with open(config_path, 'r') as f:
            self.config = yaml.load(f.read(), Loader=yaml.FullLoader)['neural_network']
            
        assert self.config['data_type'] in ['img', 'mel', 'audio', 'multi_modal'], \
            f'data_type: {self.config["data_type"]} not in available options'

        self.genre_dict = {
            0:'blues',
            1:'classical',
            2:'country',
            3:'disco',
            4:'hiphop',
            5:'jazz',
            6:'metal',
            7:'pop',
            8:'reggae',
            9:'rock',
        }

        torch.manual_seed(self.config['seed'])
        # Paramters
        self.epochs = self.config['epochs']
        self.log_step = self.config['log_step']
        self.val_check_n_epochs = self.config['val_check_n_epochs']
        logging.info(f'Using config: {config_path}')
        logging.info('Recording training logs to directory: logs/')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load and split data
        logging.info('Loading Datasets..')
        self.train_dl, self.val_dl, self.test_dl, in_d = load_dataloaders(self.config['data_type'], 
                                                                          self.config['batch_size'],
                                                                          self.config['use_n_seconds'])
        logging.info('Loading Model..')
        if self.config['finetune']:
            self.model = pretrained_resnet50('IMAGENET1K_V2').to(self.device)
        else:
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
        running_train_metrics = self._init_metrics_dict()
        running_val_metrics = self._init_metrics_dict()
        for e in range(self.epochs):
            # Validation
            val_metrics = self.eval_iter(self.val_dl)
            running_val_metrics = self._add_metrics(val_metrics, running_val_metrics)
            # Train 
            train_metrics = self.train_iter(e)
            running_train_metrics = self._add_metrics(train_metrics, running_train_metrics)
            # Reduce LR & Log Using last val_acc
            self.scheduler.step(running_val_metrics['acc'][-1])
            self._log_metrics(e+1, running_train_metrics, running_val_metrics, self.get_lr())
        # Test
        testset_metrics = self.eval_iter(self.test_dl, True)
        logging.info(f'Testset Accuracy: {testset_metrics["acc"]}')
        # Save training run
        folder_name = f'{e+1}-{self.config["data_type"]}-{self.config["architechture"]}'
        os.makedirs(f'logs/{folder_name}', exist_ok=True)
        torch.save(self.model.state_dict(), f'logs/{folder_name}/model.ckpt')
        torch.save(running_train_metrics, f'logs/{folder_name}/train_metrics.pt')
        torch.save(running_val_metrics, f'logs/{folder_name}/val_metrics.pt')
        with open(f'logs/{folder_name}/config.json', 'w') as f:
            f.write(json.dumps({
                'model_parameters': self.count_parameters(),
                'architechture': self.config['architechture'],
                'epochs': e+1,
                'batch_size': self.config['batch_size'],
                'testset_metrics':testset_metrics,
            }))   

    def train_iter(self, epoch):
        train_progress_bar = tqdm(self.train_dl, 
                                  f'Epoch: {epoch}/{self.epochs}', 
                                  leave=False)
        metrics = {'loss':0, 'acc':0, 'precision':0, 'recall':0}
        for i, data in enumerate(train_progress_bar):
            inputs, targets = self.to_device(data)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            metrics['loss'] += loss.item()
            metrics = self.calc_metrics(outputs, targets, metrics)
        metrics = self._normalise_metrics(metrics, len(self.train_dl))
        return metrics

    def eval_iter(self, dataloader, plot_confusion=False):
        metrics = {'loss':0, 'acc':0, 'precision':0, 'recall':0}
        outputs, targets = [],[]
        with torch.no_grad():
            for data in dataloader:
                x, y = self.to_device(data)
                y_pred = self.model(x)
                metrics['loss'] += self.criterion(y_pred, y).item()
                metrics = self.calc_metrics(y_pred, y, metrics)
                if plot_confusion:
                    outputs.append(torch.max(y_pred, dim=1)[1])
                    targets.append(y)
        # Metrics
        metrics = self._normalise_metrics(metrics, len(dataloader))
        if plot_confusion:
            outputs = [self.genre_dict[x] for x in torch.cat(outputs).flatten().detach().cpu().tolist()]
            targets = [self.genre_dict[x] for x in torch.cat(targets).flatten().detach().cpu().tolist()]
            metrics['cf_matrix'] = confusion_matrix(targets, outputs, labels=list(self.genre_dict.values())).tolist()
            return metrics
        else:
            return metrics

    def calc_metrics(self, outputs, targets, metrics):
        targets = targets.detach().cpu()
        outputs = outputs.detach().cpu()
        # Accuracy
        _, outputs = torch.max(outputs, dim=1)
        correct_vals = torch.sum(outputs == targets)
        total_vals = outputs.shape[0]
        metrics['acc'] += ((correct_vals / total_vals) * 100).item()
        metrics['precision'] += precision_score(targets, 
                                                outputs, 
                                                average='macro', 
                                                zero_division=True)
        metrics['recall'] += recall_score(targets, 
                                          outputs, 
                                          average='macro', 
                                          zero_division=True)
        return metrics

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
            model = ResNet2d(in_d=3, n_layers=n_layers, n_classes=10)
        elif data_type == 'multi_modal':
            model = MultiModalNet(n_layers)
        if load_trained_model != '' and load_trained_model != None:
            model.load_state_dict(torch.load(load_trained_model))
        return model

    def _normalise_metrics(self, metrics:dict, dataloader_len:int):
        metrics['loss'] /= dataloader_len
        metrics['acc'] /= dataloader_len
        metrics['precision'] /= dataloader_len
        metrics['recall'] /= dataloader_len
        return metrics

    def _add_metrics(self, metrics:dict, running_metrics:dict):
        running_metrics['loss'].append(metrics['loss'])
        running_metrics['acc'].append(metrics['acc'])
        running_metrics['precision'].append(metrics['precision'])
        running_metrics['recall'].append(metrics['recall'])
        return running_metrics

    def _init_metrics_dict(self, metrics=['loss','acc','precision','recall']):
        return {metric: [] for metric in metrics}

    def _log_metrics(self, epoch, running_train_metrics, running_val_metrics, lr):
        self.metrics_table.add_row([
            epoch,
            round(running_train_metrics['loss'][-1], 5), 
            f'{round(running_train_metrics["acc"][-1], 2)}%', 
            round(running_val_metrics['loss'][-1], 5), 
            f'{round(running_val_metrics["acc"][-1], 2)}%', 
            round(lr, 8)])
        print( "\n".join(self.metrics_table.get_string().splitlines()[-2:])) # Print only new row
