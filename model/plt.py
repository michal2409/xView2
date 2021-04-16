import os
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from apex.optimizers import FusedAdam, FusedNovoGrad, FusedSGD
from dllogger import JSONStreamBackend, Logger, StdOutBackend, Verbosity
from PIL import Image
from scipy.ndimage.morphology import binary_fill_holes
from torch_optimizer import AdaBelief, AdaBound, AdamP, RAdam
from utils.f1 import F1
from utils.scheduler import NoamLR

from model.loss import Loss
from model.unet import UNetLoc, get_dmg_unet


class Model(pl.LightningModule):
    def __init__(self, args):
        super(Model, self).__init__()
        self.save_hyperparameters()
        self.args = args
        self.f1_score = F1(args)
        self.model = UNetLoc(args) if args.type == "pre" else get_dmg_unet(args)
        self.loss = Loss(args)
        self.best_f1 = torch.tensor(0)
        self.best_epoch = 0
        self.tta_flips = [[2], [3], [2, 3]]
        self.lr = args.lr
        self.n_class = 2 if self.args.type == "pre" else 5
        self.softmax = nn.Softmax(dim=1)
        self.dllogger = Logger(
            backends=[
                JSONStreamBackend(Verbosity.VERBOSE, os.path.join(args.results, f"{args.logname}.json")),
                StdOutBackend(Verbosity.VERBOSE, step_format=lambda step: f"Epoch: {step} "),
            ]
        )

    def forward(self, img):
        pred = self.model(img)
        if self.args.tta:
            for flip_idx in self.tta_flips:
                pred += self.flip(self.model(self.flip(img, flip_idx)), flip_idx)
            pred /= len(self.tta_flips) + 1
        return pred

    def training_step(self, batch, _):
        img, lbl = batch["image"], batch["mask"]
        pred = self.model(img)
        loss = self.compute_loss(pred, lbl)
        return loss

    def validation_step(self, batch, _):
        img, lbl = batch["image"], batch["mask"]
        pred = self.forward(img)
        loss = self.loss(pred, lbl)
        self.f1_score.update(pred, lbl)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        img, lbl = batch["image"], batch["mask"]
        pred = self.forward(img)
        self.f1_score.update(pred, lbl)
        self.save_imgs(pred, lbl, batch_idx)

    def compute_loss(self, preds, label):
        if self.args.deep_supervision:
            loss = self.loss(preds[0], label)
            for i, pred in enumerate(preds[1:]):
                downsampled_label = torch.nn.functional.interpolate(label.unsqueeze(1), pred.shape[2:])
                loss += 0.5 ** (i + 1) * self.loss(pred, downsampled_label.squeeze(1))
            c_norm = 1 / (2 - 2 ** (-len(preds)))
            return c_norm * loss
        return self.loss(preds, label)

    @staticmethod
    def metric_mean(name, outputs):
        return torch.stack([out[name] for out in outputs]).mean(dim=0)

    @staticmethod
    def update_damage_scores(metrics, dmgs_f1):
        if dmgs_f1 is not None:
            for i in range(4):
                metrics.update({f"D{i+1}": round(dmgs_f1[i].item(), 3)})

    def on_validation_epoch_start(self):
        self.f1_score.reset()

    def on_test_epoch_start(self):
        self.f1_score.reset()

    def validation_epoch_end(self, outputs):
        loss = self.metric_mean("val_loss", outputs)
        f1_score, dmgs_f1 = self.f1_score.compute()
        self.f1_score.reset()

        if f1_score >= self.best_f1:
            self.best_f1 = f1_score
            self.best_epoch = self.current_epoch

        if int(os.getenv("LOCAL_RANK", "0")) == 0:
            metrics = {
                "f1": round(f1_score.item(), 3),
                "val_loss": round(loss.item(), 3),
                "top_f1": round(self.best_f1.item(), 3),
            }
            self.update_damage_scores(metrics, dmgs_f1)
            self.dllogger.log(step=self.current_epoch, data=metrics)
            self.dllogger.flush()

        self.log("f1_score", f1_score.cpu())
        self.log("val_loss", loss.cpu())

    def test_epoch_end(self, _):
        f1_score, dmgs_f1 = self.f1_score.compute()
        self.f1_score.reset()
        if int(os.getenv("LOCAL_RANK", "0")) == 0:
            metrics = {"f1": round(f1_score.item(), 3)}
            self.update_damage_scores(metrics, dmgs_f1)
            self.dllogger.log(step=(), data=metrics)
            self.dllogger.flush()

    def save_imgs(self, probs, targets, batch_idx):
        preds = torch.argmax(probs, dim=1)
        for i, (prob, pred, target) in enumerate(zip(probs, preds, targets)):
            prob = prob.cpu().detach().numpy()
            pred, target = self.to_numpy(pred), self.to_numpy(target)
            task = "localization" if self.args.type == "pre" else "damage"
            if self.args.type == "pre":
                pred = binary_fill_holes(pred).astype(np.uint8)
            idx = self.args.val_batch_size * batch_idx + i
            np.save(os.path.join("/results/predictions", f"test_{task}_{idx:05d}_probs"), prob)
            self.save(pred, "predictions", f"test_{task}_{idx:05d}_prediction.png")
            self.save(target, "targets", f"test_{task}_{idx:05d}_target.png")

    @staticmethod
    def to_numpy(tensor):
        return tensor.cpu().detach().numpy().astype(np.uint8)

    def save(self, img, mode, fname):
        Image.fromarray(img).save(os.path.join(self.args.results, mode, fname))

    def on_test_epoch_start(self):
        self.f1_score.reset()

    @staticmethod
    def flip(data, axis):
        return torch.flip(data, dims=axis)

    def configure_optimizers(self):
        optimizer = {
            "sgd": FusedSGD(self.parameters(), lr=self.lr, momentum=self.args.momentum),
            "adam": FusedAdam(self.parameters(), lr=self.lr, weight_decay=self.args.weight_decay),
            "adamw": torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.args.weight_decay),
            "radam": RAdam(self.parameters(), lr=self.lr, weight_decay=self.args.weight_decay),
            "adabelief": AdaBelief(self.parameters(), lr=self.lr, weight_decay=self.args.weight_decay),
            "adabound": AdaBound(self.parameters(), lr=self.lr, weight_decay=self.args.weight_decay),
            "adamp": AdamP(self.parameters(), lr=self.lr, weight_decay=self.args.weight_decay),
            "novograd": FusedNovoGrad(self.parameters(), lr=self.lr, weight_decay=self.args.weight_decay),
        }[self.args.optimizer.lower()]

        if not self.args.use_scheduler:
            return optimizer

        scheduler = {
            "scheduler": NoamLR(
                optimizer=optimizer,
                warmup_epochs=self.args.warmup,
                total_epochs=self.args.epochs,
                steps_per_epoch=len(self.train_dataloader()) // self.args.gpus,
                init_lr=self.args.init_lr,
                max_lr=self.args.lr,
                final_lr=self.args.final_lr,
            ),
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        arg = parser.add_argument
        arg("--optimizer", type=str, default="adamw", help="Optimizer")
        arg("--dmg_model", type=str, default="siamese", help="U-Net variant for damage assessment task")
        arg("--encoder", type=str, default="resnest200", help="U-Net encoder")
        arg("--loss_str", type=str, default="focal+dice", help="String used for creation of loss function")
        arg("--init_lr", type=float, default=1e-4, help="initial learning rate for scheduler")
        arg("--lr", type=float, default=3e-4, help="learning rate (or target learning rate for scheduler)")
        arg("--final_lr", type=float, default=1e-4, help="final learning rate for scheduler")
        arg("--weight_decay", type=float, default=1e-4, help="weight decay")
        arg("--momentum", type=float, default=0.9, help="momentum for SGD")
        arg("--dilation", type=int, choices=[1, 2, 4], default=1, help="Dilation rate for encoder")
        arg("--tta", action="store_true", help="Enable test time augmentation")
        arg("--use_scheduler", action="store_true", help="Enable learning rate scheduler")
        arg("--warmup", type=int, default=1, help="Warmup epochs for learning rate scheduler")
        arg("--ppm", action="store_true", help="Use pyramid pooling module")
        arg("--aspp", action="store_true", help="Use atrous spatial pyramid pooling")
        arg("--no_skip", action="store_true", help="Disable skip connections in UNet")
        arg("--deep_supervision", action="store_true", help="Enable deep supervision")
        arg("--attention", action="store_true", help="Enable attention module at the decoder")
        arg("--autoaugment", action="store_true", help="Use imageNet autoaugment pipeline")
        arg("--interpolate", action="store_true", help="Interpolate feature map from encoder without decoder")
        arg("--dec_interp", action="store_true", help="Use interpolation instead of transposed convolution in decoder")
        return parser