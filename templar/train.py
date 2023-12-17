import torch
from torch.utils.data import DataLoader

import timm

import os
import yaml
import argparse
import pandas as pd
import glob

from tqdm import tqdm
from importlib import import_module
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# custom import 
from utils import seed_everything
from losses import create_criterion
from optimizers import create_optimizer
from augmentation import *
from datasets import *

import warnings 
warnings.filterwarnings('ignore')


def parser_arguments():
    """
    parser 설정
    """
    parser = argparse.ArgumentParser(description="Train Config File.")
    parser.add_argument("--config", type=str, default="./config/base.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    return args

def load_yaml(config_path):
    """
    config_path의 yaml 파일 불러오기
    """
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device):

    print('Training Start ...')
    print(f'Training Device : {device} ')
    print('='*50)

    # check save path
    save_path = os.path.join(config['save'], config['model']['model_name'], config['train']['exp'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # config save 
    with open(os.path.join(save_path, config['train']['exp'] + '.yaml'), "w") as file:
        yaml.dump(config, file)

    # train loops
    epochs = config['train']['epochs']
    best_val_metric = 0
    
    # early stopping 
    early_stopping = config['setting']['early_stopping']
    if early_stopping:
        early_stopping_patience = config['setting']['patience']
        counter = 0

    for epoch in range(epochs):
        model.train()

        train_loss = 0.0
        all_labels = []
        all_preds = []

        count = 0 # for middle loss

        for idx, (imgs, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            # middle loss 
            count += 1
            if count % config['train']['middle_term'] == 0:
                print(f'Middle loss : {train_loss / count:.4f}')

        epoch_loss = train_loss / len(train_loader)
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')
        epoch_acc = accuracy_score(all_labels, all_preds)


        print(f'Training Loss: {epoch_loss:.4f} | F1 Score: {epoch_f1:.4f} | Accuracy: {epoch_acc:.4f}')

        scheduler.step() # update scheduler

        #val
        model.eval()
        all_labels_val = []
        all_preds_val = []

        with torch.no_grad():
            for inputs_val, labels_val in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}/{epochs}', leave=False):
                inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)

                # Forward pass
                outputs_val = model(inputs_val)

                # Store predictions and labels for F1 score and accuracy computation
                _, preds_val = torch.max(outputs_val, 1)
                all_labels_val.extend(labels_val.cpu().numpy())
                all_preds_val.extend(preds_val.cpu().numpy())

        epoch_f1_val = f1_score(all_labels_val, all_preds_val, average='macro')
        epoch_acc_val = accuracy_score(all_labels_val, all_preds_val)

        print(f'Validation F1 Score: {epoch_f1_val:.4f} | Accuracy: {epoch_acc_val:.4f}')

        if epoch_f1_val > best_val_metric:  
                best_val_metric = epoch_f1_val
                torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}_{epoch_f1_val:.4f}.pth'))
                print(f'Save Best Model {best_val_metric:.4f}')
                counter = 0
        else:
            if early_stopping:
                counter += 1
                if counter >= early_stopping_patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

def run_train():
    print('='*50)
    print('Config loading ...')
    # args setting
    args = parser_arguments()

    # load config yaml
    config = load_yaml(args.config)
    print('='*50)
    print(config)
    print('='*50)

    # fix seed
    seed_num = config['train']['seed']
    print(f'Seed Fix : {seed_num}')
    seed_everything(seed_num)
    print('='*50)

    # device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # data
    if config['data']['split']:
        df = pd.read_csv(config['data']['train_csv'])
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'])
    
    else:
        train_df = pd.read_csv(config['data']['train_csv'])
        val_df = pd.read_csv(config['data']['val_csv'])


    print(f'Train len : {len(train_df)}')
    print(f'Val len : {len(val_df)}')

    #if your df has not 'path', 'label' columns -> (if other names) you must input 'info = {'path' : your image path columns, 'label' : your label columns}
    train_dataset = CustomDataset(train_df, BaseAug(resize=config['train']['augmentation']['input_size']))
    val_dataset = CustomDataset(val_df, BaseAug(resize=config['val']['augmentation']['input_size']))
    
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'])
    val_loader = DataLoader(val_dataset, batch_size=config['val']['batch_size'])

    model = timm.create_model(
                            config['model']['model_name'],
                            pretrained=False,
                            num_classes=config['model']['classes_num']
                            ).to(device)

    # loss function & optimizer & scheduler
    criterion = create_criterion(config['train']['loss'])
    optimizer = create_optimizer(
        config['train']['optimizer'],
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = config['train']['lr']
    )

    schedule_module = getattr(import_module("torch.optim.lr_scheduler"), config['train']['scheduler']['name'])
    scheduler =  schedule_module(
        optimizer,
        step_size=config['train']['scheduler']['step_size'],
        gamma=config['train']['scheduler']['gamma'])
    
    # training 
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device)
