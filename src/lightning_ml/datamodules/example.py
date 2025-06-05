from __future__ import annotations

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from . import DATAMODULE_REG

__all__ = ["RandomDataModule"]


@DATAMODULE_REG.register("RandomDataModule")
class RandomDataModule(LightningDataModule):
    """Simple DataModule generating random data for demo purposes."""

    def __init__(self, input_dim: int = 32, length: int = 1024, batch_size: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.length = length
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str | None = None) -> None:
        data = torch.randn(self.length, self.input_dim)
        targets = torch.randint(0, 2, (self.length,))
        dataset = TensorDataset(data, targets)
        self.train_dataset = dataset
        self.val_dataset = dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
