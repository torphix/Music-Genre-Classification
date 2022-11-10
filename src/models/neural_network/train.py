import yaml
import torch
import logging
import torch.nn as nn
from tqdm import tqdm
from .data import Dataset
from .model import ResNet
from .scheduler import CustomScheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split


class Trainer:
    def __init__(self):
        with open('config.yaml', 'r') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)['neural_network']
            
        torch.manual_seed(config['seed'])
        # Paramters
        self.epochs = config['epochs']
        self.log_step = config['log_step']

        logging.info('Recording training logs to directory: logs/')
        self.logger = SummaryWriter('logs')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load and split data
        dataset = Dataset('data')
        val_size = int(config['val_split_size'] * dataset.__len__())
        train_size = int(dataset.__len__() - val_size)
        logging.info(f'Splitting dataset: train_size: {train_size}, val_size: {val_size}')
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        self.train_dataloader = DataLoader(train_dataset, config['batch_size'], True, collate_fn=dataset.collate_fn)
        self.val_dataloader = DataLoader(val_dataset, config['batch_size'], True, collate_fn=dataset.collate_fn)

        logging.info('Loading Model..')
        # Instantiate model (128 input features 10 possible output classes)
        model = ResNet(in_d=128, out_d=10, n_blocks=config['n_blocks'])
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=config['optim']['learning_rate'], 
                                    momentum=config['optim']['learning_rate'])
        self.scheduler = CustomScheduler(self.optimizer, **config['scheduler'])
        logging.info(f'Starting training for: {config["epochs"]}')

    def __call__(self):
        self.val_loss = self.val_iter()
        start_val_loss = self.val_loss
        for e in range(self.epochs):
            # Validation
            if e % self.log_step == 0:
                self.val_loss = self.val_iter()
            train_loss = self.train_iter(e)
        
        final_val_loss = self.val_iter()
        logging.info(f'Training Complete. Start Val loss: {start_val_loss} Final Val loss: {final_val_loss}')

    def train_iter(self, epoch):
        train_progress_bar = tqdm(self.train_dataloader, 'Train Epoch')
        for i, data in enumerate(train_progress_bar):
            mels, targets = data['mels'].to(self.device), data['targets'].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(mels)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            if i % self.log_step == 0:
                train_progress_bar.set_description(
                    f'Epoch: {epoch}/{self.epochs}, \
                      Train Loss: {round(loss.item(), 4)} \
                      Val Loss: {round(self.val_loss.item(), 4)}')
    
    def val_iter(self):
        logging.info('Running Validation..')
        val_loss = 0
        for data in self.val_dataloader:
            with torch.no_grad():
                mels, targets = data['mels'].to(self.device), data['targets'].to(self.device)
                outputs = self.model(mels)
                val_loss += self.criterion(outputs, targets)
        val_loss /= self.val_dataloader.__len__()
        return val_loss