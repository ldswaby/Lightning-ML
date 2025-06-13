import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection._split import BaseCrossValidator
from torch.utils.data import DataLoader, Dataset, Subset

from ..utils.inspect import get_class_parameters
from .dataset import BaseDataset


class DataModule(LightningDataModule, ABC):
    """
    A LightningDataModule that can wrap *any* BaseDataset subclass.
    """

    splits = {"train", "val", "test"}

    def __init__(self, **dataloader_kwargs):
        super().__init__()
        self._dataloader_kwargs = dataloader_kwargs
        self.datasets: Dict[str, BaseDataset] = {}

    def prepare_data(self) -> None:
        """Download data (executed once per node by Lightning)"""

    @abstractmethod
    def define_datasets(self) -> None:
        """Create train/val/test ``Dataset`` objects and cache them on ``self.
        datasets``.
        """
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        """Create train/val/test ``Dataset`` objects and cache them on ``self.
        datasets``.
        """
        self.define_datasets()
        if not set(self.datasets.keys()) <= self.splits:
            raise KeyError(f"self.dataset keys must within {self.splits}.")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["train"], shuffle=True, **self._dataloader_kwargs
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if "val" in self.datasets:
            return DataLoader(self.datasets["val"], **self._dataloader_kwargs)

    def test_dataloader(self) -> Optional[DataLoader]:
        if "test" in self.datasets:
            return DataLoader(self.datasets["test"], **self._dataloader_kwargs)
