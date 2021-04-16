import os

from pytorch_lightning import LightningDataModule

from data_loading.pytorch_loader import fetch_pytorch_loader


class DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_path = os.path.join(args.data, "train")
        self.val_path = os.path.join(args.data, "test")
        self.test_path = os.path.join(args.data, "holdout")

        self.train_loader_kwargs = {
            "batch_size": self.args.batch_size,
            "pin_memory": True,
            "num_workers": self.args.num_workers,
            "drop_last": True,
            "shuffle": True,
        }
        self.test_loader_kwargs = {
            "batch_size": self.args.val_batch_size,
            "pin_memory": True,
            "num_workers": self.args.num_workers,
            "drop_last": False,
            "shuffle": False,
        }

    def train_dataloader(self):
        return fetch_pytorch_loader(
            self.train_path, self.args.type, True, self.train_loader_kwargs, self.args.autoaugment
        )

    def val_dataloader(self):
        return fetch_pytorch_loader(self.val_path, self.args.type, False, self.test_loader_kwargs)

    def test_dataloader(self):
        return fetch_pytorch_loader(self.test_path, self.args.type, False, self.test_loader_kwargs)
