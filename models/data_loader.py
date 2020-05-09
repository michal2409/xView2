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
        self.mean = np.load(os.path.join('/results', 'michalf', 'xview2', 'models', 'mean.npy'))
        self.pre_imgs = sorted(glob.glob(os.path.join(data_dir, 'images/*pre*')))
        self.pre_masks = sorted(glob.glob(os.path.join(data_dir, 'masks/*pre*')))
        assert len(self.pre_imgs) == len(self.pre_masks)

        self.damage = damage
        if self.damage:
            self.post_imgs = sorted(glob.glob(os.path.join(data_dir, 'images/*post*')))        
            self.post_masks = sorted(glob.glob(os.path.join(data_dir, 'masks/*post*')))
            assert len(self.post_imgs) == len(self.post_masks) and len(self.pre_imgs) == len(self.post_imgs)

        self.hflip = HorizontalFlip(p=1)
        self.vflip = VerticalFlip(p=1)
        self.scale = RandomScale(scale_limit=0.3, p=1)
        self.crop = CropNonEmptyMaskIfExists(512, 512, p=1) # For data balancing
        self.transpose = Transpose(p=1)
        self.rotate = RandomRotate90(p=1)
        self.mask_dropout = MaskDropout(p=1)

    def __len__(self):
        return len(self.pre_imgs)

    def __getitem__(self, idx):
        img = (cv2.imread(self.pre_imgs[idx])-self.mean) / 255
        mask = cv2.imread(self.pre_masks[idx], cv2.IMREAD_UNCHANGED)
        
        if self.damage:
            post_img = (cv2.imread(self.post_imgs[idx])-self.mean) / 255
            img = np.concatenate([img, post_img], axis=2)
            post_mask = cv2.imread(self.post_masks[idx], cv2.IMREAD_UNCHANGED)
            masks = [np.expand_dims((post_mask == i).astype(int), axis = 2) for i in [1,2,3,4]]
            masks = [np.expand_dims((post_mask > 0).astype(int), axis = 2)] + masks + [np.expand_dims(post_mask, axis=2)]
            mask = np.concatenate(masks, axis=2)

        aug = self.crop(image=img, mask=mask)
        
        img = torch.from_numpy(aug['image'].transpose((2, 0, 1))).float()
        mask = torch.from_numpy(aug['mask']).long()
        return img, mask

#     def _zoom(self, img, mask, scale):
# #         res = random.random() * 0.6 + 0.7
# #         aug['image'], aug['mask'] = self._zoom(aug['image'], aug['mask'], res)        
#         height, width = img.shape[:2]
#         new_height, new_width = int(height * scale), int(width * scale)
#         img, mask = cv2.resize(img, (new_height, new_width)), cv2.resize(mask, (new_height, new_width))
#         return img, mask

class ValidationDataset(Dataset):
    def __init__(self, data_dir, damage):
        self.toTensor = T.ToTensor()
        self.mean = np.load(os.path.join('/results', 'michalf', 'xview2', 'models', 'mean.npy'))
        self.pre_imgs = sorted(glob.glob(os.path.join(data_dir, 'images/*pre*')))
        self.pre_masks = sorted(glob.glob(os.path.join(data_dir, 'masks/*pre*')))
        
        self.damage = damage
        if self.damage:
            self.post_imgs = sorted(glob.glob(os.path.join(data_dir, 'images/*post*')))        
            self.post_masks = sorted(glob.glob(os.path.join(data_dir, 'masks/*post*')))
            assert len(self.post_imgs) == len(self.post_masks) and len(self.pre_imgs) == len(self.post_imgs)

    def __len__(self):
        return len(self.pre_imgs)

    def __getitem__(self, idx):
        img = (cv2.imread(self.pre_imgs[idx])-self.mean) / 255
        mask = cv2.imread(self.pre_masks[idx], cv2.IMREAD_UNCHANGED)
        
        if self.damage:
            post_img = (cv2.imread(self.post_imgs[idx])-self.mean) / 255
            img = np.concatenate([img, post_img], axis=2)
            post_mask = cv2.imread(self.post_masks[idx], cv2.IMREAD_UNCHANGED)
            masks = [np.expand_dims((post_mask == i).astype(int), axis = 2) for i in [1,2,3,4]]
            masks = [np.expand_dims((post_mask > 0).astype(int), axis = 2)] + masks + [np.expand_dims(post_mask, axis=2)]
            mask = np.concatenate(masks, axis=2)
            
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        mask = torch.from_numpy(mask).long()
        return img, mask

def fetch_dataloader(data_dir, batch_size, dataset, damage):
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 16, 'pin_memory': True}
    if dataset == 'Training':
        return DataLoader(TrainingDataset(data_dir, damage), **kwargs)
    return DataLoader(ValidationDataset(data_dir, damage), drop_last=True, **kwargs)

