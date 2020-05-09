import torch
import logging
import json
import random
import numpy as np
import os
import sys
import datetime

DAMAGE = False

def get_damage_loss(pred, true, loss_fn):
    loss0 = loss_fn(pred[:, 0, ...], true[:, :, :, 0].type_as(pred))
    loss1 = loss_fn(pred[:, 1, ...], true[:, :, :, 1].type_as(pred))
    loss2 = loss_fn(pred[:, 2, ...], true[:, :, :, 2].type_as(pred))
    loss3 = loss_fn(pred[:, 3, ...], true[:, :, :, 3].type_as(pred))
    loss4 = loss_fn(pred[:, 4, ...], true[:, :, :, 4].type_as(pred))
    loss = 0.05 * loss0 + 0.2 * loss1 + 0.8 * loss2 + 0.7 * loss3 + 0.4 * loss4
    return loss

class RunningF1():
    def __init__(self):
        self.TP, self.FP, self.FN = 0, 0, 0
        self.TP_damage = [0,0,0,0]
        self.FN_damage = [0,0,0,0]
        self.FP_damage = [0,0,0,0]        

    def precision(self, TP, FP):
        assert TP >= 0 and FP >= 0
        if TP == 0: return 0
        else: return TP / (TP+FP)

    def recall(self, TP, FN):
        assert TP >= 0 and FN >= 0
        if TP == 0: return 0
        return TP / (TP+FN)
    
    def f1(self, prec, rec):
        assert 0 <= prec <= 1 and 0 <= rec <= 1
        if prec == 0 or rec == 0: return 0
        return (2*prec*rec) / (prec+rec)
    
    def get_damage_f1(self, c):
        prec = self.precision(self.TP_damage[c-1], self.FN_damage[c-1])
        rec = self.recall(self.TP_damage[c-1], self.FP_damage[c-1])
        return self.f1(prec, rec)
        
    def update(self, pred, targ):
        if DAMAGE:
            for c in [1,2,3,4]: 
                self.TP_damage[c-1] += np.logical_and(pred == c, targ == c).sum()
                self.FN_damage[c-1] += np.logical_and(pred != c, targ == c).sum()
                self.FP_damage[c-1] += np.logical_and(pred == c, targ != c).sum()
        else:
            self.TP += np.logical_and(pred == 1, targ == 1).sum()
            self.FN += np.logical_and(pred != 1, targ == 1).sum()
            self.FP += np.logical_and(pred == 1, targ != 1).sum()

    def __call__(self):
        if DAMAGE:
            df1 = [self.get_damage_f1(c) for c in [1,2,3,4]]
            print(df1)
            return len(df1) / sum((x+1e-6)**-1 for x in df1)
            
        prec, rec = self.precision(self.TP, self.FP), self.recall(self.TP, self.FN)
        assert 0 <= prec <= 1 and 0 <= rec <= 1
        if prec == 0 or rec == 0:
            return 0
        return self.f1(prec, rec)

class ConfigParser():
    def __init__(self, filename):
        self.settings = {}

        with open(filename, "r") as f:
            config = json.load(f)

        self.config = config
        self.read_field_value("train_data", "/scidatasm/michalf/train")
        self.read_field_value("val_data", "/scidatasm/michalf/hold")
        self.read_field_value("batch_size", 64)
        self.read_field_value("model_name", "Unet")
        self.read_field_value("pretrained_model", None)
        self.read_field_value("lr", 1e-3)
        self.read_field_value("epochs", 1)

        self.read_field_value("hflip", 0)
        self.read_field_value("vflip", 0)
        self.read_field_value("scale", 0)
        self.read_field_value("transpose", 0)
        self.read_field_value("rotate", 0)
        self.read_field_value("mask_dropout", 0)
        self.read_field_value("test_time_augmentation", 0)

    def read_field_value(self, key, default):
        param = default
        if key in self.config:
            param = self.config[key]
        self.settings[key] = param

    def get_parameters(self):
        return self.settings

class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)

def log_epoch(epoch, n_epochs, train_loss, train_f1, val_loss, val_f1):
        logging.info(f'Epoch: [{epoch+1}/{n_epochs}] loss: {train_loss:.4f} - f1: {train_f1:.4f} - val_loss: {val_loss:.4f} - val_f1 {val_f1:.4f}')

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(os.path.join(log_path, 'train.log'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
        
def get_output_dir(test=False):
    path = os.path.join('/results', 'michalf')
    path = os.path.join(path, 'predictions') if test else os.path.join(path, 'experiments')
    dirname = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    path = os.path.join(path, dirname)
    os.makedirs(path)
    return path

def set_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic=True
    
def prepare_env(test=False):
    device = torch.device('cuda:0')
    config = ConfigParser('params.json')
    params = config.get_parameters()
    output_dir = get_output_dir(test)
    set_logger(output_dir)
    
    return device, params, output_dir

def load_model(model, path):
    pretrained_model = torch.load(path)
    for name, tensor in pretrained_model.items():
        name = name.replace('module.', '', 1)
        model.state_dict()[name].copy_(tensor)
    return model
