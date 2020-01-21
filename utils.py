import torch
import logging
import json
import random
import numpy as np
import os

class RunningF1():
    def __init__(self):
        self.score = 0.5
        self.score_avg = 0.5
        self.TP, self.FP, self.FN = 0, 0, 0
        self.P, self.R = 0, 0
    
    def precision(self):
        assert self.TP >= 0 and self.FP >= 0
        if self.TP == 0: return 0
        else: return self.TP/(self.TP+self.FP)

    def recall(self):
        assert self.TP >= 0 and self.FN >= 0
        if self.TP == 0: return 0
        return self.TP/(self.TP+self.FN)

    def f1(self):
        assert 0 <= self.P <= 1 and 0 <= self.R <= 1
        if self.P == 0 or self.R == 0: return 0
        return (2*self.P*self.R)/(self.P+self.R)
    
    def update(self, pred, targ):
        self.TP += np.logical_and(pred == 1, targ == 1).sum()
        self.FN += np.logical_and(pred != 1, targ == 1).sum()
        self.FP += np.logical_and(pred == 1, targ != 1).sum()
        
        self.P = self.precision()
        self.R = self.recall()
        self.score = self.f1()
        self.score_avg = 0.9*self.score_avg + 0.1*self.score
        
    def get_avg(self):
        return self.score_avg
    
    def __call__(self):
        return self.score

class ConfigParser():
    def __init__(self, filename):
        self.settings = {}
        
        with open(filename, "r") as f:
            config = json.load(f)

        self.config = config
        self.read_field_value("train_data", "data/datasets/train_dataset")
        self.read_field_value("val_data", "data/datasets/val_dataset")
        self.read_field_value("batch_size", 64)
        self.read_field_value("model_name", "Unet")       
        self.read_field_value("pretrained_model", None)
        self.read_field_value("epochs", 1)
        self.read_field_value("lr", 1e-3)
        self.read_field_value("epochs", 1)
        
        self.read_field_value("hflip", 1)
        self.read_field_value("vflip", 1)
        self.read_field_value("scale", 1)
        self.read_field_value("transpose", 1)
        self.read_field_value("rotate", 1)
        self.read_field_value("mask_dropout", 1)
        self.read_field_value("test_time_augmentation", 1)
        
        
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
    
def log_epoch(epoch, n_epochs, train_loss, train_acc, val_loss, val_acc, val_f1):
        logging.info(f'Epoch: [{epoch+1}/{n_epochs}] loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc {val_acc:.4f} - val_f1 {val_f1:.4f}')
    
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
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
        
def get_output_dir():
    assert os.path.isdir("experiments")
    i = 0
    while os.path.exists("experiments/%s" % i):
        i += 1
    os.mkdir("experiments/%s" % i)
    return "experiments/%s" % i

def set_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic=True