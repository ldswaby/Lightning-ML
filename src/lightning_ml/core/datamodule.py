import inspect
from abc import ABC, abstractmethod
from types import SimpleNamespace
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

    def __init__(self):
        super().__init__()
        self._dataloader_kwargs = {}
        self.datasets: Dict[str, BaseDataset] = {}

    @abstractmethod
    def define_datasets(self) -> None:
        """Create train/val/test ``Dataset`` objects and cache them on ``self.
        datasets``.
        """
        raise NotImplementedError

    # def _logic(self, stage: str):
    #     """TODO: this is a placeholder function, but all data splitting logic should happen within define_Dataset

    #     this logic can be defined
    #     depending e.g. on which script is run (train/test).

    #     Args:
    #         stage (str): _description_
    #     """
    #     if stage == "fit":
    #         # if dataset class has a `train` flag, invoke that = True, alng
    #         # with other args
    #         # e.g. mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
    #         # if doens't have train flag...
    #         pass
    #     if stage == "test":
    #         # train=True
    #         pass
    #     if stage == "predict":
    #         # train=True
    #         pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Create train/val/test ``Dataset`` objects and cache them on ``self.
        datasets``.
        """
        self.define_datasets()
        if not self.datasets:
            raise ValueError("`self.datasets` is empty.")
        if not set(self.datasets.keys()) <= self.splits:
            raise KeyError(f"self.dataset keys must be within {self.splits}.")

    @property
    def dataloader_kwargs(self):
        return self._dataloader_kwargs

    @dataloader_kwargs.setter
    def dataloader_kwargs(self, kwargs: Dict[str, Any]) -> None:
        # Validate that provided kwargs are valid DataLoader parameters
        valid_params = inspect.signature(DataLoader).parameters
        invalid_keys = set(kwargs) - set(valid_params)
        if invalid_keys:
            raise ValueError(f"Invalid DataLoader kwargs: {invalid_keys}")
        self._dataloader_kwargs = kwargs

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["train"], shuffle=True, **self.dataloader_kwargs
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if "val" in self.datasets:
            return DataLoader(self.datasets["val"], **self.dataloader_kwargs)

    def test_dataloader(self) -> Optional[DataLoader]:
        if "test" in self.datasets:
            return DataLoader(self.datasets["test"], **self.dataloader_kwargs)

    @classmethod
    def from_config(cls, cfg: dict) -> "DataModule":
        # TODO
        raise NotImplementedError
