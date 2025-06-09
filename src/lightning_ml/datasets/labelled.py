"""
Standard dataset implementation based on LabelledDatasetBase.

This module provides a simple in-memory labeled dataset that stores
input and target sequences.
"""

from typing import Any, Callable, Optional, Sequence

from .unlabelled import UnlabelledDataset

__all__ = ["LabelledDataset"]


class LabelledDataset(UnlabelledDataset):
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
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        if len(inputs) != len(targets):
            raise ValueError("inputs and targets must have the same length")

        super().__init__(inputs, transform=transform)

        self._targets = targets
        self.target_transform = target_transform or (lambda y: y)

    def get_target(self, idx: int) -> Any:
        """Raw (un-transformed) target lookup."""
        return self.target_transform(self._targets[idx])
