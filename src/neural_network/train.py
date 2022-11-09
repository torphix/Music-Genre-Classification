import yaml
import torch
import logging
import torch.nn as nn
from tqdm import tqdm
from .data import Dataset
from .model import ResNet
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

def train():

    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    logging.info('Recording training logs to directory: logs/')
    logger = SummaryWriter('logs')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load and split data
    dataset = Dataset('data')
    val_size = int(config['val_split_size'] * dataset.__len__())
    train_size = int(dataset.__len__() - val_size)
    logging.info(f'Splitting dataset: train_size: {train_size}, val_size: {val_size}')
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, config['batch_size'], True, collate_fn=dataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, config['batch_size'], True, collate_fn=dataset.collate_fn)

    logging.info('Loading Model..')
    # Instantiate model (128 input features 10 possible output classes)
    model = ResNet(in_d=128, out_d=10, n_blocks=4)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=config['optim']['learning_rate'], 
                                momentum=config['optim']['learning_rate'])

    logging.info(f'Starting training for: {config["epochs"]}')

    val_loss = 0
    for e in range(config['epochs']):
        train_progress_bar = tqdm(train_dataloader, 'Train Epoch')

        # Validation
        if e % config['val_check_n_epochs'] == 0 or val_loss == 0:
            logging.info('Running Validation..')
            for data in val_dataloader:
                with torch.no_grad():
                    mels, targets = data['mels'].to(device), data['targets'].to(device)
                    outputs = model(mels)
                    val_loss = criterion(outputs, targets)

        for i, data in enumerate(train_progress_bar):
            mels, targets = data['mels'].to(device), data['targets'].to(device)
            optimizer.zero_grad()
            outputs = model(mels)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if i % config['log_step'] == 0:
                train_progress_bar.set_description(f'Epoch: {e}/{config["epochs"]}, Train Loss: {round(loss.item(), 4)} Val Loss: {round(val_loss.item(), 4)}')
    
    logging.info('Training Complete')