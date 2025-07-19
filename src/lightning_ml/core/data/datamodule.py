"""Generic LightningDataModule with optional cross-validation support."""

from __future__ import annotations

from typing import Any, Dict, Optional, Type
from collections.abc import Callable

import pytorch_lightning as pl
from sklearn.model_selection._split import BaseCrossValidator
from torch.nn import ModuleDict
from torch.utils.data import DataLoader

from ...utils.data import validation_split
from .dataset import BaseDataset

__all__ = ["DataModule"]


class DataModule(pl.LightningDataModule):
    """Create data loaders from a dataset with optional CV splits.

    This will be a base class that will produce subclasses (e.g. ImageClassification) with various classmethods e.g.

    TODO:
    1. If just train, split off test, then split off val (see README.md)
    2. If train and test, split off val

    """

    def __init__(
        self,
        train_data: BaseDataset | None,
        val_data: BaseDataset | None = None,
        test_data: BaseDataset | None = None,
        predict_data: BaseDataset | None = None,
        val_split: float | None = None,
        test_split: float | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        **dataloader_kwargs,
    ) -> None:
        super().__init__()

        self._train_data = train_data
        self._val_data = val_data
        self._test_data = test_data
        self._predict_data = predict_data

        # Arg checks
        if (
            self._train_data
            and self._test_data
            and isinstance(test_split, float)
            and test_split > 0
        ):
            raise TypeError(
                "A `test_dataset` was provided with `test_split`. Please, choose one or the other."
            )

        if (
            self._train_data
            and self._val_data
            and isinstance(val_split, float)
            and val_split > 0
        ):
            raise TypeError(
                "A `val_dataset` was provided with `val_split`. Please, choose one or the other."
            )

        self.val_split = val_split
        self.transform = transform
        self.target_transform = target_transform
        self.dataloader_kwargs = dataloader_kwargs

    @property
    def train_dataset(self) -> BaseDataset | None:
        """This property returns the train dataset."""
        return self._train_data

    @property
    def val_dataset(self) -> BaseDataset | None:
        """This property returns the validation dataset."""
        return self._val_data

    @property
    def test_dataset(self) -> BaseDataset | None:
        """This property returns the test dataset."""
        return self._test_data

    @property
    def predict_dataset(self) -> BaseDataset | None:
        """This property returns the prediction dataset."""
        return self._predict_data

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader | None:
        if self.val_dataset is not None:
            return DataLoader(self.val_dataset, **self.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader | None:
        if self.test_dataset is not None:
            return DataLoader(self.test_dataset, **self.dataloader_kwargs)

    @classmethod
    def from_config(cls, cfg: dict) -> DataModule:
        # TODO
        raise NotImplementedError
