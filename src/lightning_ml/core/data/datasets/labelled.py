"""
Standard dataset implementation based on LabelledDatasetBase.

This module provides a simple in-memory labeled dataset that stores
input and target sequences.
"""

from collections.abc import Callable, Sequence
from typing import Any

from ...utils.enums import Registries
from ...utils.registry import register
from ..dataset import LabelledDatasetBase

__all__ = ["LabelledDataset"]


@register(Registries.DATASET)
class LabelledDataset(LabelledDatasetBase):
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
        targets: Sequence[Any],
        *,
        transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[Any], Any] | None = None,
    ) -> None:
        if len(inputs) != len(targets):
            raise ValueError("inputs and targets must have the same length")

        # TODO: create some kind of InputLoader (either arg or Mixin) that can load data here from various formats, e.g. Image, ImageFiles, ImageFolder
        # The sole purpose of the inputs classes is to fetch different data types from disk and load them into Sequence[Any] for the attributes below

        self._inputs = inputs
        self._targets = targets
        self.transform = transform or (lambda x: x)
        self.target_transform = target_transform or (lambda y: y)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self._inputs)

    @property
    def targets(self) -> Sequence[Any]:
        """Return all targets"""
        return self._targets

    def get_input(self, idx: int) -> Any:
        """Transformed input lookup."""
        return self.transform(self._inputs[idx])

    def get_target(self, idx: int) -> Any:
        """Raw (un-transformed) target lookup."""
        return self.target_transform(self._targets[idx])
