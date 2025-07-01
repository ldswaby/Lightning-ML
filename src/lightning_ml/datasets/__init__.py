"""Dataset registry and built-in dataset classes."""

from __future__ import annotations

from ..utils.registry import Registry

DATASET_REG = Registry("Dataset")

from .abstract import *  # noqa: F401,F403
from .contrastive import *  # noqa: F401,F403
from .labelled import *  # noqa: F401,F403
from .disk import *  # noqa: F401,F403
from .torchvision_wrapper import *  # noqa: F401,F403

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
