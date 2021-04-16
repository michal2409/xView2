import os
import random
from glob import glob

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import PIL
import torch
from torch.utils.data import DataLoader, Dataset

from data_loading.autoaugment import ImageNetPolicy


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def fetch_pytorch_loader(path, mode, training, loader_kwargs, autoaugment=False):
    if not training:
        dataset = TestDataset
    elif mode == "pre":
        dataset = TrainPreDataset
    else:
        dataset = TrainPostDataset
    return DataLoader(dataset(path, mode, autoaugment), worker_init_fn=seed_worker, **loader_kwargs)


def load_data(path, dtype):
    imgs = sorted(glob(os.path.join(path, "images", f"*{dtype}*")))
    lbls = sorted(glob(os.path.join(path, "targets", f"*{dtype}*")))
    assert len(imgs) == len(lbls) and len(imgs) > 0
    return imgs, lbls


def load_pair(img, lbl):
    img = cv2.imread(img)
    lbl = cv2.imread(lbl, cv2.IMREAD_UNCHANGED)
    return img, lbl


def intensity_aug(aug, img1, img2=None):
    img1 = aug(image=img1)["image"]
    if img2 is None:
        return img1
    img2 = aug(image=img2)["image"]
    return img1, img2


class TrainPreDataset(Dataset):
    def __init__(self, path, _, autoaugment):
        self.imgs_pre, self.lbls_pre = load_data(path, "pre")
        assert len(self.imgs_pre) == len(self.lbls_pre)
        self.crop = A.CropNonEmptyMaskIfExists(p=1, width=512, height=512)
        self.zoom = A.RandomScale(p=0.2, scale_limit=(0, 0.3), interpolation=cv2.INTER_CUBIC)
        self.hflip = A.HorizontalFlip(p=0.33)
        self.vflip = A.VerticalFlip(p=0.33)
        self.noise = A.GaussNoise(p=0.1)
        self.brctr = A.RandomBrightnessContrast(p=0.2)
        self.normalize = A.Normalize()
        data_frame = pd.read_csv("/workspace/xview2/utils/index.csv")
        self.idx = data_frame["idx"].tolist()
        self.use_autoaugment = autoaugment
        if self.use_autoaugment:
            self.autoaugment = ImageNetPolicy()

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        img, lbl = load_pair(self.imgs_pre[self.idx[idx]], self.lbls_pre[self.idx[idx]])
        data = {"image": img, "mask": lbl}
        if not self.use_autoaugment:
            data = self.zoom(image=data["image"], mask=data["mask"])
        data = self.crop(image=data["image"], mask=data["mask"])
        if self.use_autoaugment:
            img = PIL.Image.fromarray(data["image"])
            lbl = PIL.Image.fromarray(data["mask"])
            img, lbl = self.autoaugment(img, lbl)
            img = np.array(img)
            data["mask"] = np.array(lbl)
        else:
            data = self.hflip(image=data["image"], mask=data["mask"])
            data = self.vflip(image=data["image"], mask=data["mask"])
            img = intensity_aug(self.noise, data["image"])
            img = intensity_aug(self.brctr, img)
        img = intensity_aug(self.normalize, img)
        data["image"] = np.transpose(img, (2, 0, 1))
        return data


class TrainPostDataset(Dataset):
    def __init__(self, path, _, autoaugment):
        self.imgs_pre, self.lbls_pre = load_data(path, "pre")
        self.imgs_post, self.lbls_post = load_data(path, "post")
        assert len(self.imgs_pre) == len(self.imgs_post)
        assert len(self.imgs_post) == len(self.lbls_post)
        data_frame = pd.read_csv("/workspace/xview2/utils/index.csv")
        self.idx = []
        self.idx.extend(data_frame[data_frame["1"] == 1]["idx"].values.tolist())
        self.idx.extend(data_frame[data_frame["2"] == 1]["idx"].values.tolist())
        self.idx.extend(data_frame[data_frame["3"] == 1]["idx"].values.tolist())
        self.idx.extend(data_frame[data_frame["4"] == 1]["idx"].values.tolist())
        self.idx = sorted(list(set(self.idx)))

        self.crop = A.CropNonEmptyMaskIfExists(p=1, width=512, height=512)
        self.zoom = A.RandomScale(p=0.2, scale_limit=(0, 0.3), interpolation=cv2.INTER_CUBIC)
        self.hflip = A.HorizontalFlip(p=0.33)
        self.vflip = A.VerticalFlip(p=0.33)
        self.noise = A.GaussNoise(p=0.1)
        self.brctr = A.RandomBrightnessContrast(p=0.2)
        self.normalize = A.Normalize()

        self.use_autoaugment = autoaugment
        if self.use_autoaugment:
            self.autoaugment = ImageNetPolicy()

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        img_pre, _ = load_pair(self.imgs_pre[self.idx[idx]], self.lbls_pre[self.idx[idx]])
        img_post, lbl = load_pair(self.imgs_post[self.idx[idx]], self.lbls_post[self.idx[idx]])
        img = np.concatenate((img_pre, img_post), axis=2)
        data = {"image": img, "mask": lbl}
        if not self.use_autoaugment:
            data = self.zoom(image=data["image"], mask=data["mask"])
        data = self.crop(image=data["image"], mask=data["mask"])
        if self.use_autoaugment:
            img_pre = PIL.Image.fromarray(data["image"][:, :, :3])
            img_post = PIL.Image.fromarray(data["image"][:, :, 3:])
            lbl = PIL.Image.fromarray(data["mask"])
            img_pre, lbl, img_post = self.autoaugment(img_pre, lbl, img_post)
            img_pre, img_post = np.array(img_pre), np.array(img_post)
            data["mask"] = np.array(lbl)
        else:
            data = self.hflip(image=data["image"], mask=data["mask"])
            data = self.vflip(image=data["image"], mask=data["mask"])
            img_pre, img_post = data["image"][:, :, :3], data["image"][:, :, 3:]
            img_pre, img_post = intensity_aug(self.noise, img_pre, img_post)
            img_pre, img_post = intensity_aug(self.brctr, img_pre, img_post)
        img_pre, img_post = intensity_aug(self.normalize, img_pre, img_post)
        data["image"] = np.concatenate((img_pre, img_post), axis=2)
        data["image"] = np.transpose(data["image"], (2, 0, 1))
        return data


class TestDataset(Dataset):
    def __init__(self, path, mode, _):
        self.mode = mode
        self.imgs_pre, self.lbls_pre = load_data(path, "pre")
        self.imgs_post, self.lbls_post = load_data(path, "post")
        self.normalize = A.Normalize()
        assert len(self.imgs_pre) == len(self.imgs_post)
        assert len(self.imgs_post) == len(self.lbls_post)

    def __len__(self):
        return len(self.imgs_pre)

    def __getitem__(self, idx):
        img, lbl = load_pair(self.imgs_pre[idx], self.lbls_pre[idx])
        img = intensity_aug(self.normalize, img)
        if self.mode == "post":
            img_post, lbl = load_pair(self.imgs_post[idx], self.lbls_post[idx])
            img_post = intensity_aug(self.normalize, img_post)
            img = np.concatenate((img, img_post), axis=2)
        img = np.transpose(img, (2, 0, 1))
        return {"image": img, "mask": lbl}
