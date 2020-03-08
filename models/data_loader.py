from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch
import os
import glob
import json
import random
import PIL
import cv2

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    RandomScale,
    Transpose,
    RandomRotate90,
    MaskDropout,
    CropNonEmptyMaskIfExists,
    RandomCrop,
    Resize
)

class TrainingDataset(Dataset):
    def __init__(self, data_dir, augm):
        self.augm = augm
        self.toTensor = T.ToTensor()
        self.mean = np.load(os.path.join('/results', 'michalf', 'xview2', 'models', 'mean.npy'))
        self.imgs = sorted(glob.glob(os.path.join(data_dir, 'images/*')))
        self.labels = sorted(glob.glob(os.path.join(data_dir, 'labels/*')))
        self.masks = sorted(glob.glob(os.path.join(data_dir, 'masks/*')))
        assert len(self.imgs) == len(self.labels) and len(self.imgs) == len(self.masks)

        self.hflip = HorizontalFlip()
        self.vflip = VerticalFlip()
        self.scale = RandomScale(scale_limit=0.3)
        self.crop = CropNonEmptyMaskIfExists(512, 512, p=1) # For data balancing
        self.transpose = Transpose()
        self.rotate = RandomRotate90()
        self.mask_dropout = MaskDropout()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = (cv2.imread(self.imgs[idx])-self.mean) / 255
        mask = cv2.imread(self.masks[idx], 0)
        
        aug = {}
        aug['image'], aug['mask'] = img, mask
        if self.augm['hflip']:
            aug = self.hflip(image=aug['image'], mask=aug['mask'])
        if self.augm['vflip']:
            aug = self.vflip(image=aug['image'], mask=aug['mask'])
        if self.augm['scale']:  
            aug = self.scale(image=aug['image'], mask=aug['mask'])
        if self.augm['transpose']: 
            aug = self.transpose(image=aug['image'], mask=aug['mask'])
        if self.augm['rotate']:
            aug = self.rotate(image=aug['image'], mask=aug['mask'])
        if self.augm['mask_dropout']:
            aug = self.mask_dropout(image=aug['image'], mask=aug['mask'])
        aug = self.crop(image=aug['image'], mask=aug['mask'])
        
        return self.toTensor(aug['image']), self.toTensor(aug['mask'])
    
class ValidationDataset(Dataset):
    def __init__(self, data_dir):
        self.toTensor = T.ToTensor()
        self.mean = np.load(os.path.join('/results', 'michalf', 'xview2', 'models', 'mean.npy'))
        self.imgs = sorted(glob.glob(os.path.join(data_dir, 'images/*')))
        self.masks = sorted(glob.glob(os.path.join(data_dir, 'masks/*')))
        assert len(self.imgs) == len(self.masks)

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.toTensor((cv2.imread(self.imgs[idx])-self.mean) / 255)
        mask = self.toTensor(cv2.imread(self.masks[idx], 0))
        return img, mask

# class TestDataset(Dataset):
#     def __init__(self, data_dir):
#         self.imgs = [i for i in imgs if i.rstrip('.png').split('_')[2] == 'pre']
#         self.idx = [i.rstrip('.png').split('_')[3] for i in self.imgs]
#         self.mean = np.load(os.path.join('/results', 'michalf', 'xview2', 'models', 'mean.npy'))
#         self.toTensor = T.ToTensor() 

#     def __len__(self):
#         return len(self.imgs)

#     def __getitem__(self, idx):
#         img = self.toTensor((cv2.imread(self.imgs[idx])-self.mean) / 255)
#         return X, self.idx[idx]

def fetch_dataloader(data_dir, batch_size, dataset, params=None):
    assert dataset in ['Training', 'Validation', 'Test']
    if dataset == 'Training':
        return DataLoader(TrainingDataset(data_dir, params), batch_size=batch_size, shuffle=True, num_workers=8)
    elif dataset == 'Validation':
        return DataLoader(ValidationDataset(data_dir), batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    return DataLoader(TestDataset(data_dir), batch_size=batch_size, shuffle=True, num_workers=8)

