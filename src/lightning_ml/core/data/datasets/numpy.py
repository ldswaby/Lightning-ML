"""Datasets backed by NumPy arrays or ``.npy`` files."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ...utils.enums import Registries
from ...utils.registry import register
from .labelled import LabelledDataset
from .unlabelled import UnlabelledDataset

__all__ = ["NumpyUnlabelledDataset", "NumpyLabelledDataset"]


@register(Registries.DATASET)
class NumpyUnlabelledDataset(UnlabelledDataset):
    """Unlabelled dataset backed by a NumPy array or ``.npy`` file."""

    def __init__(self, inputs: str | Sequence[Any] | np.ndarray, *, transform: Callable[[Any], Any] | None = None) -> None:
        if isinstance(inputs, str):
            inputs = np.load(inputs)
        super().__init__(inputs, transform=transform)


@register(Registries.DATASET)
class NumpyLabelledDataset(LabelledDataset):
    """Labelled dataset backed by NumPy arrays or ``.npy`` files."""

    def __init__(
        self,
        inputs: str | Sequence[Any] | np.ndarray,
        targets: str | Sequence[Any] | np.ndarray,
        *,
        transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[Any], Any] | None = None,
    ) -> None:
        if isinstance(inputs, str):
            inputs = np.load(inputs)
        if isinstance(targets, str):
            targets = np.load(targets)
        super().__init__(inputs, targets, transform=transform, target_transform=target_transform)
