import os
import numpy as np
import torch.nn as nn
import torch
import json
import sys
import random
import utils
import timeit
import logging
import matplotlib.pyplot as plt
import models.data_loader as data_loader
from models.models import *
from PIL import Image
from parallel import *

DAMAGE = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def _save_imgs(y_pred, y_true, idx, output_dir):
    for i in range(y_pred.shape[0]):
        pred, true = np.array(y_pred[i]).astype(np.uint8), np.array(y_true[i]).astype(np.uint8)
        Image.fromarray(pred).save(os.path.join(output_dir, 'prediction', f"test_localization_{idx+i:05d}_prediction.png"))
        Image.fromarray(pred).save(os.path.join(output_dir, 'prediction', f"test_damage_{idx+i:05d}_prediction.png"))
        if y_true is not None:
            Image.fromarray(true).save(os.path.join(output_dir, 'target', f"test_localization_{idx+i:05d}_target.png"))
            Image.fromarray(true).save(os.path.join(output_dir, 'target', f"test_damage_{idx+i:05d}_target.png"))
        
def _test_time_augmentation(model, X, y_pred, device):
    y_vflip = torch.flip(model(torch.flip(X, [2]).to(device).float()), [2])
    y_hflip = torch.flip(model(torch.flip(X, [3]).to(device).float()), [3])
    y_pred = (y_pred+y_hflip+y_vflip) / 3
    return y_pred

def evaluate(model, loss_fn, dataloader, device, save=False, output_dir=None):
    model.eval()
    f1 = utils.RunningF1()
    with torch.no_grad():
        for i, (X, y_true) in enumerate(dataloader):
            X, y_true = X.to(device), y_true.to(device)
            y_pred = model(X)
            
            if DAMAGE:
                _, predicted = torch.max(y_pred.data, 1)
            else:
                predicted = (y_pred[:, 0] > 0).long()
            f1.update(predicted.squeeze().cpu().numpy(), y_true.squeeze().cpu().numpy())
            if save:
                _save_imgs(predicted.cpu().numpy(), y_true.cpu().numpy(), i*X.shape[0], output_dir)
    return f1()

if __name__ == '__main__':
    utils.set_seed()
    device, params, output_dir = utils.prepare_env(test=True)
    os.makedirs(os.path.join(output_dir, 'prediction'))
    os.makedirs(os.path.join(output_dir, 'target'))

    logging.info("Fetching dataloader...")
    dataloader = data_loader.fetch_dataloader("/scidatasm/michalf/test", params['batch_size'], 'Validation', DAMAGE)

    logging.info("Buidling model...")
    if DAMAGE:
        model = Res101_Unet_Double().to(device)
    else:
        model = Res101_Unet_Loc().to(device)
        
    logging.info("Finished building...")
    if params['load_model'] is not None:
        logging.info("Loading pretrained weights")
        model = utils.load_model(model, params['load_model'])
    model = nn.DataParallel(model)
    logging.info("Evaluation...")
    f1 = evaluate(model, dataloader, device, params, False, output_dir)
    logging.info(f'F1 {f1:.4f}')
    logging.info("Finished...")
