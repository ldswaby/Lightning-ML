from typing import Any, Optional

import numpy as np
import torch

from .abstract import SupervisedDataset


class NumpyDataset(SupervisedDataset):
    """
    A supervised dataset that loads X.npy and y.npy arrays from disk.
    """

    def __init__(
        self,
        x_path: str,
        y_path: str,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ):
        self.X = np.load(x_path)
        self.y = np.load(y_path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.X)

    def get_data(self, idx: int) -> torch.Tensor:
        x = torch.from_numpy(self.X[idx])
        return self.transform(x) if self.transform else x

    def get_target(self, idx: int) -> torch.Tensor:
        t = torch.from_numpy(self.y[idx])
        return self.target_transform(t) if self.target_transform else t
