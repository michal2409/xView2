import models.data_loader as data_loader
import numpy as np
import torch.nn as nn
import torch
import os
import json
import sys
import random
import utils
import logging
from tqdm import tqdm
import segmentation_models_pytorch as smp
from models.deeplab import DeepLab
import loss
import torchvision.transforms.functional as TF

utils.set_seed()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
config = utils.ConfigParser('params.json')
params = config.get_parameters()

# Preparing directory for results
output_dir = utils.get_output_dir()
utils.set_logger(output_dir)
with open(os.path.join(output_dir, 'params.json'), 'w') as outfile:
    json.dump(params, outfile)

logging.info("Fetching dataloaders...")
train_loader = data_loader.fetch_dataloader(params['train_data'], params['batch_size'], 'Training', params)
val_loader = data_loader.fetch_dataloader(params['val_data'], params['batch_size'], 'Validation')

logging.info("Buidling model...")
# model = DeepLab(num_classes=2)
# model = PSPNet()
model = smp.Unet('resnet101', classes=2, encoder_weights='imagenet', activation='softmax').to(device)
logging.info("Finished building...")

if params['pretrained_model'] is not None:
    logging.info("Loading pretrained weights")
    pretrained_model = torch.load(params['pretrained_model'])
    for name, tensor in pretrained_model.items():
        name = name.replace('module.', '', 1)
        model.state_dict()[name].copy_(tensor)
        
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[0,1,2])

def update_pbar(y_pred, y_true, acc, f1, loss_avg, t):
    _, predicted = torch.max(y_pred.data, 1)
    correct, total = (predicted == y_true).sum().item(), y_true.shape[0]*y_true.shape[1]*y_true.shape[2]
    acc.update(100*correct/total)
    f1.update(predicted.squeeze().cpu().numpy(), y_true.squeeze().cpu().numpy())
    t.set_postfix_str(f'acc={acc():.4f} f1={f1.get_avg():.4f} loss={loss_avg():.4f}')
    t.update()

def train_epoch():
    model.train()
    total, correct = 0, 0
    train_acc, train_loss_avg, f1_tr = utils.RunningAverage(), utils.RunningAverage(), utils.RunningF1()
    with tqdm(total=len(train_loader)) as t:
        for i, (X, y_true) in enumerate(train_loader):
            X, y_true = X.to(device), y_true.type(torch.LongTensor).to(device).squeeze()
            optimizer.zero_grad()
            y_pred = model(X.float())
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            optimizer.step()
            train_loss_avg.update(loss.item())
            update_pbar(y_pred, y_true, train_acc, f1_tr, train_loss_avg, t)
    scheduler.step()
    return train_acc(), train_loss_avg()

def save_model(name):
    logging.info("Saving model!")
    torch.save(model.state_dict(), os.path.join(output_dir, name))

    
# Dodaj rotacje do test time augmentation 
def rotate(image, mask, angle):
    image = TF.rotate(image, angle)
    segmentation = TF.rotate(segmentation, angle)
    return image, segmentation


def evaluate():
    model.eval()
    total, correct = 0, 0
    val_acc, val_loss_avg, f1_val = utils.RunningAverage(), utils.RunningAverage(), utils.RunningF1()

    with tqdm(total=len(val_loader)) as t:
        with torch.no_grad():
            for i, (X, y_true) in enumerate(val_loader):
                X, y_true = X.to(device).float(), y_true.type(torch.LongTensor).to(device).squeeze()
                y_pred = model(X)
                if params['test_time_augmentation']:
                    X_vflip = torch.flip(X, [2]).to(device).float()
                    X_hflip = torch.flip(X, [3]).to(device).float()
                    y_vflip = torch.flip(model(X_vflip), [2])
                    y_hflip = torch.flip(model(X_hflip), [3])
                    # Taking mean of predictions
                    y_pred += y_hflip+y_vflip
                    y_pred /= 3
                
                loss = loss_fn(y_pred, y_true)
                val_loss_avg.update(loss.item())
                update_pbar(y_pred, y_true, val_acc, f1_val, val_loss_avg, t)
    return val_acc(), val_loss_avg(), f1_val.get_avg()

loss_fn = torch.nn.CrossEntropyLoss()
# loss_fn = loss.JaccardLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 8], gamma=0.1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, eta_min=1e-7)

n_epochs = params['epochs']
best_val_loss, best_f1 = float('Inf'), 0.0
for epoch in range(n_epochs):
    train_acc, train_loss = train_epoch()
    val_acc, val_loss, f1_val = evaluate()
    utils.log_epoch(epoch, n_epochs, train_loss, train_acc, val_loss, val_acc, f1_val)

    if best_val_loss > val_loss:
        best_val_loss = val_loss
        save_model('best_loss.pth')
        
    if best_f1 < f1_val:
        best_f1 = f1_val
        save_model('best_f1.pth')
        
save_model('last.pth')
