import os
import numpy as np
import torch.nn as nn
import torch
import json
import sys
import random
import utils
import logging
import segm as smp
import matplotlib.pyplot as plt
import models.data_loader as data_loader

def _save_imgs(y_pred, y_true, idx, output_dir):
    for i in range(y_pred.shape[0]):
        plt.imsave(os.path.join(output_dir, str(idx+i)+'pred.png'), y_pred[i])
        if y_true is not None:
            plt.imsave(os.path.join(output_dir, str(idx+i)+'true.png'), y_true.squeeze()[i])
        
def _test_time_augmentation(X, y_pred, device):
    y_vflip = torch.flip(model(torch.flip(X, [2]).to(device).float()), [2])
    y_hflip = torch.flip(model(torch.flip(X, [3]).to(device).float()), [3])
    y_pred = (y_pred+y_hflip+y_vflip) / 3
    
    return y_pred

def evaluate(model, dataloader, device, params, save=False, output_dir=None):
    model.eval()
    total, correct = 0, 0
    loss_fn = torch.nn.CrossEntropyLoss()
    acc, loss_avg, f1 = utils.RunningAverage(), utils.RunningAverage(), utils.RunningF1()

    with torch.no_grad():
        for i, (X, y_true) in enumerate(dataloader):
            X, y_true = X.to(device).float(), y_true.type(torch.LongTensor).to(device).squeeze()
            y_pred = model(X)
            if params['test_time_augmentation']:
                y_pred = _test_time_augmentation(X, y_pred, device)

            loss = loss_fn(y_pred, y_true)
            loss_avg.update(loss.item())
            
            _, predicted = torch.max(y_pred.data, 1)
            correct, total = (predicted == y_true).sum().item(), y_true.shape[0]*y_true.shape[1]*y_true.shape[2]
            acc.update(100*correct/total)
            f1.update(predicted.squeeze().cpu().numpy(), y_true.squeeze().cpu().numpy())
            if save:
                _save_imgs(predicted.cpu(), y_true.cpu(), i*X.shape[0], output_dir)
        
    return acc(), loss_avg(), f1()

# def test(model, dataloader, device, params, output_dir):
#     model.eval()
#     with torch.no_grad():
#         for X in dataloader:
#             X = X.to(device).float()
#             y_pred = model(X)
#             if params['test_time_augmentation']:
#                 y_pred = _test_time_augmentation(X, y_pred, device)
#             _save_imgs(predicted.cpu(), None, i*X.shape[0], output_dir)

if __name__ == '__main__':
    utils.set_seed()
    device, params, output_dir = utils.prepare_env(test=True)

    logging.info("Fetching dataloader...")
    dataloader = data_loader.fetch_dataloader(params['val_data'], params['batch_size'], 'Validation')

    logging.info("Buidling model...")
    model = smp.Unet('resnet101', classes=2, encoder_weights='imagenet', activation='softmax').to(device)
    logging.info("Finished building...")

    if params['pretrained_model'] is not None:
        logging.info("Loading pretrained weights")
        model = utils.load_model(model, params['pretrained_model'])

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    logging.info("Evaluation...")
    acc, loss, f1 = evaluate(model, dataloader, device, params, True, output_dir)
    logging.info(f'Loss: {loss:.4f} - Acc {acc:.4f} - F1 {f1:.4f}')
    logging.info("Finished...")
