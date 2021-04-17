import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob
from subprocess import call

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from data_loading.data_module import DataModule
from model.plt import Model
from utils.gpu_affinity import set_affinity


def make_empty_dir(path):
    call(["rm", "-rf", path])
    os.makedirs(path)


def set_cuda_devices(gpus):
    assert gpus <= torch.cuda.device_count(), f"Requested {args.gpus} gpus, available {torch.cuda.device_count()}."
    device_list = ",".join([str(i) for i in range(gpus)])
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", device_list)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument
    arg("--exec_mode", type=str, choices=["train", "eval"], default="train", help="Execution mode of main script")
    arg("--data", type=str, default="/data", help="Path to the data directory")
    arg("--results", type=str, default="/results", help="Path to the results directory")
    arg("--gpus", type=int, default=1, choices=list(range(torch.cuda.device_count() + 1)), help="Number of gpus to use")
    arg("--num_workers", type=int, default=8, help="Number of subprocesses to use for data loading")
    arg("--batch_size", type=int, default=16, help="Training batch size")
    arg("--val_batch_size", type=int, default=13, help="Evaluation batch size")
    arg("--precision", type=int, default=16, choices=[16, 32], help="Numerical precision")
    arg("--epochs", type=int, default=250, help="Max number of epochs")
    arg("--patience", type=int, default=100, help="Early stopping patience")
    arg("--ckpt", type=str, default=None, help="Path to pretrained checkpoint")
    arg("--logname", type=str, default="logs", help="Name of logging file")
    arg(
        "--ckpt_pre",
        type=str,
        default=None,
        help="Path to pretrained checkpoint of localization model used to initialize network for damage assesment",
    )
    arg(
        "--type",
        type=str,
        choices=["pre", "post"],
        help="Type of task to run; pre - localization, post - damage assesment",
    )
    arg("--seed", type=int, default=1)

    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()
    if args.interpolate:
        args.deep_supervision = False
        args.dec_interp = False

    set_cuda_devices(args.gpus)
    affinity = set_affinity(os.getenv("LOCAL_RANK", "0"), "socket_unique_interleaved")
    seed_everything(args.seed)
    data_module = DataModule(args)

    callbacks = None
    checkpoint = args.ckpt if args.ckpt is not None and os.path.exists(args.ckpt) else None
    if args.exec_mode == "train":
        model = Model(args)
        model_ckpt = ModelCheckpoint(monitor="f1_score", mode="max", save_last=True)
        callbacks = [EarlyStopping(monitor="f1_score", patience=args.patience, verbose=True, mode="max")]
    else:
        assert args.ckpt is not None, "No checkpoint found for evaluation"
        model = Model.load_from_checkpoint(args.ckpt)

    if args.type == "post" and args.ckpt_pre is not None:
        pretrained_model = torch.load(args.ckpt_pre, map_location={"cuda:0": "cpu"})["state_dict"]
        keys = model.state_dict()
        for name, tensor in pretrained_model.items():
            if "enc" not in name:
                continue
            if "parallel" in args.dmg_model:
                key1, key2 = name.replace("unet", "unet_pre"), name.replace("unet", "unet_post")
                if key1 in keys:
                    model.state_dict()[key1].copy_(tensor)
                if key2 in keys:
                    model.state_dict()[keys].copy_(tensor)
            elif args.dmg_model == "siameseEnc":
                key = name.replace(".unet", "")
                if key in keys:
                    model.state_dict()[key].copy_(tensor)
            else:
                if name in keys:
                    model.state_dict()[name].copy_(tensor)

    trainer = Trainer(
        gpus=args.gpus,
        logger=False,
        precision=args.precision,
        benchmark=True,
        deterministic=False,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        max_epochs=args.epochs,
        min_epochs=args.epochs,
        sync_batchnorm=args.gpus > 1,
        accelerator="ddp" if args.gpus > 1 else None,
        default_root_dir=args.results,
        checkpoint_callback=model_ckpt if args.exec_mode == "train" else None,
        resume_from_checkpoint=checkpoint,
    )

    if args.exec_mode == "train":
        trainer.fit(model, data_module)
    else:
        pred_dir = os.path.join(args.results, "probs")
        targets_dir = os.path.join(args.results, "targets")
        if not os.path.exists(pred_dir):
            make_empty_dir(pred_dir)
        if not os.path.exists(targets_dir):
            make_empty_dir(targets_dir)
        trainer.test(model, test_dataloaders=data_module.test_dataloader())

    if "PL_TRAINER_GPUS" in os.environ:
        os.environ.pop("PL_TRAINER_GPUS")
