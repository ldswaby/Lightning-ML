"""Dataset registry and built-in dataset classes."""

from __future__ import annotations

from ...utils.registry import Registry
from .abstract import *  # noqa: F401,F403
from .contrastive import *  # noqa: F401,F403
from .disk import *  # noqa: F401,F403
from .labelled import *  # noqa: F401,F403
from .unlabelled import *  # noqa: F401,F403
from .wrappers import TorchvisionDataset

DATASET_REG = Registry("Dataset")


__all__ = [
    "DATASET_REG",
    "LabelledDataset",
    "UnlabelledDataset",
    "NumpyUnlabelledDataset",
    "NumpyLabelledDataset",
    "CSVDataset",
    "ImageFolderDataset",
    "ContrastiveLabelledDataset",
    "ContrastiveUnlabelledDataset",
    "TripletDataset",
    "TorchvisionDataset",
]
