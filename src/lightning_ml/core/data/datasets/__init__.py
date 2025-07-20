"""Dataset registry and built-in dataset classes."""

from __future__ import annotations

from lightning_ml.utils.registry import Registry

# Define the registry before importing submodules that register with it to
# avoid import cycles during package initialisation.
DATASET_REG = Registry("Dataset")

# Import dataset implementations which will register themselves with
# ``DATASET_REG`` on import.
from .abstract import *  # noqa: F401,F403
from .contrastive import *  # noqa: F401,F403
from .disk import *  # noqa: F401,F403
from .labelled import *  # noqa: F401,F403
from .unlabelled import *  # noqa: F401,F403
from .wrappers import TorchvisionDataset

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
