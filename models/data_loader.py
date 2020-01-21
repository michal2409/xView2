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
    RandomCrop
)

class TrainingDataset(Dataset):
    def __init__(self, data_dir, augm):
        self.augm = augm
        self.toTensor = T.ToTensor()
        self.mean = np.load("/home/michalf/xview/models/mean.npy")
        self.imgs = sorted(glob.glob(os.path.join(data_dir, 'images/*')))
        self.labels = sorted(glob.glob(os.path.join(data_dir, 'labels/*')))
        self.masks = sorted(glob.glob(os.path.join(data_dir, 'masks/*')))
        assert len(self.imgs) == len(self.labels) and len(self.imgs) == len(self.masks)
        
        # Augmentations
        self.hflip = HorizontalFlip(p=0.5)
        self.vflip = VerticalFlip(p=0.5)
        self.scale = RandomScale(p=0.5, scale_limit=0.3)
        self.crop = CropNonEmptyMaskIfExists(512, 512, p=1) # For data balancing
        self.transpose = Transpose(p=0.5)
        self.rotate = RandomRotate90(p=0.5)
        self.mask_dropout = MaskDropout(p=0.5)

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
    
#     def random_color_distort(self,
#             img,
#             brightness_delta=32,
#             contrast_low=0.5, contrast_high=1.5,
#             saturation_low=0.5, saturation_high=1.5,
#             hue_delta=18):

#         cv_img = img[::-1].astype(np.uint8) # RGB to BGR

#         def convert(img, alpha=1, beta=0):
#             img = img.astype(float) * alpha + beta
#             img[img < 0] = 0
#             img[img > 255] = 255
#             return img.astype(np.uint8)

#         def brightness(cv_img, delta):
#             if random.randrange(2):
#                 return convert(
#                     cv_img,
#                     beta=random.uniform(-delta, delta))
#             else:
#                 return cv_img

#         def contrast(cv_img, low, high):
#             if random.randrange(2):
#                 return convert(
#                     cv_img,
#                     alpha=random.uniform(low, high))
#             else:
#                 return cv_img

#         def saturation(cv_img, low, high):
#             if random.randrange(2):
#                 cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
#                 cv_img[:, :, 1] = convert(
#                     cv_img[:, :, 1],
#                     alpha=random.uniform(low, high))
#                 return cv2.cvtColor(cv_img, cv2.COLOR_HSV2BGR)
#             else:
#                 return cv_img

#         def hue(cv_img, delta):
#             if random.randrange(2):
#                 cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
#                 cv_img[:, :, 0] = (
#                     cv_img[:, :, 0].astype(int) +
#                     random.randint(-delta, delta)) % 180
#                 return cv2.cvtColor(cv_img, cv2.COLOR_HSV2BGR)
#             else:
#                 return cv_img

#         cv_img = brightness(cv_img, brightness_delta)

#         if random.randrange(2):
#             cv_img = contrast(cv_img, contrast_low, contrast_high)
#             cv_img = saturation(cv_img, saturation_low, saturation_high)
#             cv_img = hue(cv_img, hue_delta)
#         else:
#             cv_img = saturation(cv_img, saturation_low, saturation_high)
#             cv_img = hue(cv_img, hue_delta)
#             cv_img = contrast(cv_img, contrast_low, contrast_high)

#         return cv_img[::-1]
    
class ValidationDataset(Dataset):
    def __init__(self, data_dir):
        self.toTensor = T.ToTensor()
        self.mean = np.load("/home/michalf/xview/models/mean.npy")
        self.imgs = sorted(glob.glob(os.path.join(data_dir, 'images/*')))
        self.masks = sorted(glob.glob(os.path.join(data_dir, 'masks/*')))
        assert len(self.imgs) == len(self.masks)

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.toTensor((cv2.imread(self.imgs[idx])-self.mean) / 255)
        mask = self.toTensor(cv2.imread(self.masks[idx], 0))
        return img, mask

class TestDataset(Dataset):
    def __init__(self, data_dir):
        imgs = glob.glob('/home/michalf/xview/data/datasets/test_dataset/images/*')
        self.imgs = [i for i in imgs if i.rstrip('.png').split('_')[2] == 'pre']
        self.idx = [i.rstrip('.png').split('_')[3] for i in self.imgs]
        self.mean = np.load("/home/michalf/xview/mean.npy")
        self.toTensor = T.ToTensor() 

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx])
        img = (img-self.mean)/255.0
        X = self.toTensor(img)
        return X, self.idx[idx]
    
def fetch_dataloader(data_dir, batch_size, dataset, params=None):
    assert dataset in ['Training', 'Validation', 'Test']
    if dataset == 'Training':
        return DataLoader(TrainingDataset(data_dir, params), batch_size=batch_size, shuffle=True, num_workers=8)
    elif dataset == 'Validation':
        return DataLoader(ValidationDataset(data_dir), batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    return DataLoader(TestDataset(data_dir), batch_size=batch_size, shuffle=True, num_workers=8)

