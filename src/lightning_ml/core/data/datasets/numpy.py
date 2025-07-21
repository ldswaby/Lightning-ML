"""Datasets backed by NumPy arrays or ``.npy`` files."""

from __future__ import annotations

import os
from typing import Any, Sequence

from ...utils.enums import Registries
from ...utils.registry import register
from .labelled import LabelledDataset

__all__ = ["NumpyLabelledDataset"]

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - fallback when numpy not available
    np = None


@register(Registries.DATASET)
class NumpyLabelledDataset(LabelledDataset):
    """Labelled dataset storing inputs and targets as NumPy arrays."""

    def __init__(
        self,
        inputs: Sequence[Any] | os.PathLike | str,
        targets: Sequence[Any] | os.PathLike | str,
        **kwargs: Any,
    ) -> None:
        if isinstance(inputs, (str, os.PathLike)):
            if np is None:
                raise ImportError("NumPy is required to load from file")
            inputs = np.load(inputs)
        if isinstance(targets, (str, os.PathLike)):
            if np is None:
                raise ImportError("NumPy is required to load from file")
            targets = np.load(targets)
        super().__init__(list(inputs), list(targets), **kwargs)

