"""
Standard dataset implementation based on LabelledDatasetBase.

This module provides a simple in-memory labeled dataset that stores
input and target sequences.
"""

from typing import Any, Optional
from collections.abc import Callable, Sequence

from ...utils.enums import Registries
from ...utils.registry import register
from .abstract import UnlabelledDatasetBase

__all__ = ["UnlabelledDataset"]


@register(Registries.DATASET)
class UnlabelledDataset(UnlabelledDatasetBase):
    """Generic labelled dataset with optional transforms.

    Args:
        inputs: Sequence of raw inputs.
        targets: Sequence of raw targets (must be same length as ``inputs``).
        transform: Optional callable applied to the input only.
        target_transform: Optional callable applied to the target only.
        joint_transform: Optional callable applied to ``(input, target)`` **after**
            the individual transforms.  Use this for paired augmentations
            (e.g. image & mask random crop that must stay aligned).

    Notes:
        * All transforms must be **callables** that return the same type they
          receive (e.g. ``torch.Tensor`` in → out).
        * If you need lazy loading from file paths, just store the paths here
          and open them inside the transforms or ``get_input``/``get_target``.
    """

    def __init__(
        self,
        inputs: Sequence[Any],
        *,
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        self._inputs = inputs
        self.transform = transform or (lambda x: x)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self._inputs)

    def get_input(self, idx: int) -> Any:
        """Transformed input lookup."""
        return self.transform(self._inputs[idx])
