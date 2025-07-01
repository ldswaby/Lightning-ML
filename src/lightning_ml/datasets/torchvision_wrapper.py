"""Wrapper classes to adapt torchvision datasets to Lightning-ML's Dataset API."""

from __future__ import annotations

from typing import Any, Type

from . import DATASET_REG
from .abstract import LabelledDatasetBase

__all__ = ["TorchvisionDataset"]

try:
    from torch.utils.data import Dataset as TorchDataset
except Exception as e:  # pragma: no cover - import dependency
    raise ImportError("PyTorch is required for TorchvisionDataset") from e


@DATASET_REG.register()
class TorchvisionDataset(LabelledDatasetBase):
    """Wrap a torchvision dataset returning ``(input, target)`` tuples."""

    def __init__(self, dataset: TorchDataset | Type[TorchDataset], **kwargs: Any) -> None:
        if isinstance(dataset, type):
            dataset = dataset(**kwargs)
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_tuple(self, idx: int) -> tuple[Any, Any]:
        sample = self.dataset[idx]
        if not isinstance(sample, tuple) or len(sample) < 2:
            raise ValueError(
                "Underlying dataset must return a tuple (input, target)"
            )
        return sample[0], sample[1]

    def get_input(self, idx: int) -> Any:
        inp, _ = self._get_tuple(idx)
        return inp

    def get_target(self, idx: int) -> Any:
        _, tgt = self._get_tuple(idx)
        return tgt
