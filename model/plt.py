import os
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex.optimizers import FusedAdam, FusedNovoGrad, FusedSGD
from dllogger import JSONStreamBackend, Logger, StdOutBackend, Verbosity
from PIL import Image
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
        self.test_idx = 0
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
        self.save(pred, lbl)

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

    def save(self, preds, targets):
        if self.args.type == "pre":
            probs = torch.sigmoid(preds[:, 1])
        else:
            if self.args.loss_str == "coral":
                probs = torch.sum(torch.sigmoid(preds) > 0.5, dim=1) + 1
            elif self.args.loss_str == "mse":
                probs = torch.round(F.relu(preds[:, 0], inplace=True)) + 1
            else:
                probs = self.softmax(preds)

        probs = probs.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy().astype(np.uint8)
        for prob, target in zip(probs, targets):
            task = "localization" if self.args.type == "pre" else "damage"
            fname = os.path.join(self.args.results, "probs", f"test_{task}_{self.test_idx:05d}")
            self.test_idx += 1
            np.save(fname, prob)
            Image.fromarray(target).save(fname.replace("probs", "targets") + "_target.png")

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
        arg(
            "--optimizer",
            type=str,
            default="adamw",
            choices=["sgd", "adam", "adamw", "radam", "adabelief", "adabound", "adamp", "novograd"],
        )
        arg(
            "--dmg_model",
            type=str,
            default="siamese",
            choices=["siamese", "siameseEnc", "fused", "fusedEnc", "parallel", "parallelEnc", "diff", "cat"],
            help="U-Net variant for damage assessment task",
        )
        arg(
            "--encoder",
            type=str,
            default="resnest200",
            choices=["resnest50", "resnest101", "resnest200", "resnest269", "resnet50", "resnet101", "resnet152"],
            help="U-Net encoder",
        )
        arg(
            "--loss_str",
            type=str,
            default="focal+dice",
            help="Combination of: dice, focal, ce, ohem, mse, coral, e.g focal+dice creates the loss function as sum of focal and dice",
        )
        arg("--use_scheduler", action="store_true", help="Enable Noam learning rate scheduler")
        arg("--warmup", type=int, default=1, help="Warmup epochs for Noam learning rate scheduler")
        arg("--init_lr", type=float, default=1e-4, help="Initial learning rate for Noam scheduler")
        arg("--final_lr", type=float, default=1e-4, help="Final learning rate for Noam scheduler")
        arg("--lr", type=float, default=3e-4, help="Learning rate, or a target learning rate for Noam scheduler")
        arg("--weight_decay", type=float, default=0, help="Weight decay (L2 penalty)")
        arg("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
        arg(
            "--dilation",
            type=int,
            choices=[1, 2, 4],
            default=1,
            help="Dilation rate for a encoder, e.g dilation=2 uses dilation instead of stride in the last encoder block",
        )
        arg("--tta", action="store_true", help="Enable test time augmentation")
        arg("--ppm", action="store_true", help="Use pyramid pooling module")
        arg("--aspp", action="store_true", help="Use atrous spatial pyramid pooling")
        arg("--no_skip", action="store_true", help="Disable skip connections in UNet")
        arg("--deep_supervision", action="store_true", help="Enable deep supervision")
        arg("--attention", action="store_true", help="Enable attention module at the decoder")
        arg("--autoaugment", action="store_true", help="Use imageNet autoaugment pipeline")
        arg("--interpolate", action="store_true", help="Interpolate feature map from encoder without a decoder")
        arg("--dec_interp", action="store_true", help="Use interpolation instead of transposed convolution in a decoder")
        return parser
