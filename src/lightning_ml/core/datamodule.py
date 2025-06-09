import inspect
from typing import Any, Dict, Optional, Type

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from lightning_ml.core import dataset

from .dataset import BaseDataset


class DataModule(LightningDataModule):
    """
    A LightningDataModule that can wrap *any* BaseDataset subclass.

    Args:
        dataset_cls: A subclass of BaseDataset to instantiate.
        dataset_kwargs: kwargs to pass to dataset_cls constructor.
        batch_size: batch size for all splits.
        num_workers: DataLoader num_workers.
        pin_memory: DataLoader pin_memory.
        splits: Optional dict of
            {"train": float, "val": float, "test": float} proportions
            that sum to 1.0, or explicit dict of filepaths.
        download: Whether to download the dataset if supported.
    """

    def __init__(
        self,
        dataset_cls: Type[BaseDataset],
        dataset_kwargs: Dict[str, Any],
        dataloader_kwargs: Dict[str, Any],
        splits: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.dataset_cls = dataset_cls
        self.dataset_kwargs = dataset_kwargs
        self.dataloader_kwargs = dataloader_kwargs
        self.download = dataset_kwargs.get("download", False)

        # splits can be proportions or concrete kwargs for each stage
        # e.g. {"train":0.8,"val":0.1,"test":0.1} or {"train":{...},"val":{...},...}
        self.splits = splits or {"train": 1.0, "val": 0.0, "test": 0.0}

        self.datasets: Dict[str, BaseDataset] = {}

    def prepare_data(self) -> None:
        # Trigger download if supported by dataset
        if self.download:
            _ = self.dataset_cls(**self.dataset_kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Instantiate train/val/test datasets.
        If splits are floats, we split a single array accordingly.
        Otherwise, splits can be dicts of kwargs for each dataset.
        """
        # If splits are proportions, we load one dataset and do a random split
        # TODO: plug in logic for more specialized splitting?
        if all(isinstance(v, float) for v in self.splits.values()):
            init_kwargs = dict(self.dataset_kwargs)
            if "download" in inspect.signature(self.dataset_cls.__init__).parameters:
                init_kwargs["download"] = False
            all_data = self.dataset_cls(**init_kwargs)
            n = len(all_data)
            lengths = [
                int(self.splits["train"] * n),
                int(self.splits.get("val", 0) * n),
                n - int(self.splits["train"] * n) - int(self.splits.get("val", 0) * n),
            ]
            train, val, test = torch.utils.data.random_split(all_data, lengths)
            self.datasets["train"] = train
            self.datasets["val"] = val
            self.datasets["test"] = test

        else:
            # assume splits is a dict of kwargs for each split
            for split, kwargs in self.splits.items():
                combined_kwargs = {**self.dataset_kwargs, **kwargs}
                if (
                    "download"
                    in inspect.signature(self.dataset_cls.__init__).parameters
                ):
                    combined_kwargs["download"] = False
                self.datasets[split] = self.dataset_cls(**combined_kwargs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["train"], shuffle=True, **self.dataloader_kwargs
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.datasets["val"], **self.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.datasets["test"], **self.dataloader_kwargs)


# from my_package.datamodules import GenericDataModule
# from my_package.datasets import NumpyDataset

# # 1) Proportional split
# dm = GenericDataModule(
#     dataset_cls=NumpyDataset,
#     dataset_kwargs={"x_path": "data/X.npy", "y_path": "data/y.npy"},
#     batch_size=64,
#     splits={"train": 0.8, "val": 0.1, "test": 0.1},
# )

# # 2) Explicit file-based split
# # (assuming you have train_X.npy / train_y.npy, etc.)
# dm = GenericDataModule(
#     dataset_cls=NumpyDataset,
#     dataset_kwargs={},
#     batch_size=32,
#     splits={
#         "train": {"x_path": "data/train_X.npy", "y_path": "data/train_y.npy"},
#         "val":   {"x_path": "data/val_X.npy",   "y_path": "data/val_y.npy"},
#         "test":  {"x_path": "data/test_X.npy",  "y_path": "data/test_y.npy"},
#     },
# )
