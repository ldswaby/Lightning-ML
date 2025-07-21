"""Alias module exposing datasets at :mod:`lightning_ml.datasets`."""

from ..core.data.datasets import REGISTRY as DATASET_REG
from ..core.data.datasets import *  # noqa: F401,F403
from ..core.data.datasets.wrappers import TorchvisionDataset

__all__ = ["DATASET_REG"] + [
    name
    for name in globals().keys()
    if not name.startswith("_") and name not in {"REGISTRY", "DATASET_REG"}
]
