"""Dataset implementations for NumPy arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, overload

from ...utils.enums import Registries
from ...utils.registry import register
from .labelled import LabelledDataset

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore

__all__ = ["NumpyLabelledDataset"]


@register(Registries.DATASET)
class NumpyLabelledDataset(LabelledDataset):
    """Labeled dataset backed by NumPy arrays or ``.npy`` files."""

    @overload
    def __init__(
        self,
        inputs: Sequence[Any],
        targets: Sequence[Any],
        *,
        transform: Any | None = None,
        target_transform: Any | None = None,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        inputs: str | Path,
        targets: str | Path,
        *,
        transform: Any | None = None,
        target_transform: Any | None = None,
    ) -> None:
        ...

    def __init__(
        self,
        inputs: Sequence[Any] | str | Path,
        targets: Sequence[Any] | str | Path,
        *,
        transform: Any | None = None,
        target_transform: Any | None = None,
    ) -> None:
        if np is not None:
            if isinstance(inputs, (str, Path)):
                inputs = np.load(str(inputs))
            if isinstance(targets, (str, Path)):
                targets = np.load(str(targets))
        super().__init__(list(inputs), list(targets), transform=transform, target_transform=target_transform)
