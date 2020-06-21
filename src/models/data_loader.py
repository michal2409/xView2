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
    def __init__(self, data_dir, damage):
        self.pre_imgs = sorted(glob.glob(os.path.join(data_dir, 'images/*pre*')))
        self.pre_masks = sorted(glob.glob(os.path.join(data_dir, 'masks/*pre*')))
        assert len(self.pre_imgs) == len(self.pre_masks)
        
        self.post_imgs = sorted(glob.glob(os.path.join(data_dir, 'images/*post*')))        
        self.post_masks = sorted(glob.glob(os.path.join(data_dir, 'masks/*post*')))
        assert len(self.post_imgs) == len(self.post_masks) and len(self.pre_imgs) == len(self.post_imgs)
        
        self.mean = np.load(os.path.join('/results', 'michalf', 'xview2', 'models', 'mean.npy'))
        self.damage = damage

        self.hflip = HorizontalFlip(p=0.5)
        self.vflip = VerticalFlip(p=0.3)
        self.scale = RandomScale(scale_limit=0.3)
        self.crop = CropNonEmptyMaskIfExists(512, 512, p=1)
        self.transpose = Transpose(p=0.3)
        self.rotate = RandomRotate90(p=0.3)

    def __len__(self):
        if self.damage:
            return len(self.post_imgs)
        return len(self.pre_imgs)

    def __getitem__(self, idx):
        img = (cv2.imread(self.pre_imgs[idx])-self.mean) / 255
        mask = cv2.imread(self.pre_masks[idx], cv2.IMREAD_UNCHANGED)
        if self.damage:
            post_img = (cv2.imread(self.post_imgs[idx]) - self.mean) / 255
            post_mask = cv2.imread(self.post_masks[idx], cv2.IMREAD_UNCHANGED)
            img = np.concatenate([img, post_img], axis=2)            
            masks = [np.expand_dims((post_mask == i).astype(int), axis = 2) for i in [1,2,3,4]]
            mask = np.expand_dims(mask, axis=2)
            masks.append(mask)
            mask = np.concatenate(masks, axis=2)
        
        aug = {}
        aug['image'], aug['mask'] = img, mask
        aug = self.crop(image=aug['image'], mask=aug['mask'])
#         aug = self.hflip(image=aug['image'], mask=aug['mask'])
        
        img = torch.from_numpy(aug['image'].transpose((2, 0, 1))).float()
        mask = torch.from_numpy(aug['mask']).long()
        return img, mask


class ValidationDataset(Dataset):
    def __init__(self, data_dir, damage):
        self.pre_imgs = sorted(glob.glob(os.path.join(data_dir, 'images/*pre*')))
        self.pre_masks = sorted(glob.glob(os.path.join(data_dir, 'masks/*pre*')))
        assert len(self.pre_imgs) == len(self.pre_masks)
        
        self.post_imgs = sorted(glob.glob(os.path.join(data_dir, 'images/*post*')))        
        self.post_masks = sorted(glob.glob(os.path.join(data_dir, 'masks/*post*')))
        assert len(self.post_imgs) == len(self.post_masks) and len(self.pre_imgs) == len(self.post_imgs)

        self.toTensor = T.ToTensor()
        self.mean = np.load(os.path.join('/results', 'michalf', 'xview2', 'models', 'mean.npy'))
        self.damage = damage

    def __len__(self):
        if self.damage:
            return len(self.post_imgs)
        return len(self.pre_imgs)

    def __getitem__(self, idx):
        img = (cv2.imread(self.pre_imgs[idx])-self.mean) / 255
        mask = cv2.imread(self.pre_masks[idx], cv2.IMREAD_UNCHANGED)
        if self.damage:
            post_img = (cv2.imread(self.post_imgs[idx]) - self.mean) / 255
            post_mask = cv2.imread(self.post_masks[idx], cv2.IMREAD_UNCHANGED)
            img = np.concatenate([img, post_img], axis=2)            
            masks = [np.expand_dims((post_mask == i).astype(int), axis = 2) for i in [1,2,3,4]]
            mask = np.expand_dims(mask, axis=2)
            masks.append(mask)
            mask = np.concatenate(masks, axis=2)    
            
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        mask = torch.from_numpy(mask).long()
        return img, mask

def fetch_dataloader(data_dir, batch_size, dataset, damage):
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 8, 'pin_memory': False}
    if dataset == 'Training':
        return DataLoader(TrainingDataset(data_dir, damage),  drop_last=True, **kwargs)
    return DataLoader(ValidationDataset(data_dir, damage), drop_last=True, **kwargs)
