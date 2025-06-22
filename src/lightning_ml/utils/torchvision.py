from __future__ import annotations

import inspect
from typing import Optional

from .registry import Registry

__all__ = ["register_torchvision_models", "register_torchvision_datasets", "register_torchvision"]


def _import_torchvision():
    try:
        import torchvision
        return torchvision
    except Exception as e:
        raise ImportError("torchvision is required for this functionality") from e


def register_torchvision_models(registry: Optional[Registry] = None) -> None:
    """Register all torchvision model functions and classes."""
    tv = _import_torchvision()
    from ..models import MODEL_REG  # local import to avoid circular dependency
    registry = registry or MODEL_REG

    for name, obj in inspect.getmembers(tv.models):
        if name.startswith("_"):
            continue
        if callable(obj):
            try:
                registry.register(name)(obj)
            except KeyError:
                pass


def register_torchvision_datasets(registry: Optional[Registry] = None) -> None:
    """Register all torchvision dataset classes."""
    tv = _import_torchvision()
    from ..datasets import DATASET_REG
    registry = registry or DATASET_REG

    try:
        from torch.utils.data import Dataset
    except Exception as e:
        raise ImportError("PyTorch is required for this functionality") from e

    for name, obj in inspect.getmembers(tv.datasets, inspect.isclass):
        if name.startswith("_"):
            continue
        if issubclass(obj, Dataset):
            try:
                registry.register(name)(obj)
            except KeyError:
                pass


def register_torchvision() -> None:
    """Register both torchvision datasets and models."""
    register_torchvision_models()
    register_torchvision_datasets()
