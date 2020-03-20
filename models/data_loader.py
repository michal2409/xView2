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
    def __init__(self, data_dir, params):
        self.augm = params
        self.augmentations = ['hflip', 'vflip', 'scale', 'transpose', 'rotate', 'mask_dropout']
        self.num_aug = sum([self.augm[x] for x in self.augmentations])
        self.toTensor = T.ToTensor()
        self.mean = np.load(os.path.join('/results', 'michalf', 'xview2', 'models', 'mean.npy'))
        self.imgs = sorted(glob.glob(os.path.join(data_dir, 'images/*')))
        self.labels = sorted(glob.glob(os.path.join(data_dir, 'labels/*')))
        self.masks = sorted(glob.glob(os.path.join(data_dir, 'masks/*')))
        assert len(self.imgs) == len(self.labels) and len(self.imgs) == len(self.masks)

        self.hflip = HorizontalFlip(p=1)
        self.vflip = VerticalFlip(p=1)
        self.scale = RandomScale(scale_limit=0.3, p=1)
        self.crop = CropNonEmptyMaskIfExists(512, 512, p=1) # For data balancing
        self.transpose = Transpose(p=1)
        self.rotate = RandomRotate90(p=1)
        self.mask_dropout = MaskDropout(p=1)
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = (cv2.imread(self.imgs[idx])-self.mean) / 255
        mask = cv2.imread(self.masks[idx], 0)
        
        y_aug = ""
        for augm_type in self.augmentations:
            img, mask, res = self._apply_augm(img, mask, augm_type, 0.5)
            y_aug += res
        aug = self.crop(image=img, mask=mask)
        
        return self.toTensor(aug['image']), (self.toTensor(aug['mask']), torch.tensor(int(y_aug, 2), dtype=torch.long))
    
    def _apply_augm(self, img, mask, augm_type, prob):
        aug = {}
        aug['image'], aug['mask'] = img, mask
        res = ""
        if self.augm[augm_type]:
            if random.random() < prob:
                aug = self.hflip(image=aug['image'], mask=aug['mask'])
                res = "1"
            else:
                res = "0"
        return aug['image'], aug['mask'], res
    
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

