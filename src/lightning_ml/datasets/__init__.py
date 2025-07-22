"""Dataset registry and built-in datasets."""

from __future__ import annotations

from ..core.utils.enums import Registries
from ..core.utils.registry import get_registry
from ..core.data.datasets import *  # noqa: F401,F403
from ..core.data.datasets.wrappers import TorchvisionDataset  # noqa: F401

DATASET_REG = get_registry(Registries.DATASET)

__all__ = ["DATASET_REG"]
