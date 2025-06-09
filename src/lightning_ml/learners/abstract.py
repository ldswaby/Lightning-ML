"""Base learner implementations for common paradigms."""

from __future__ import annotations

from typing import Any, Dict

from torch import Tensor

from ..core import Learner

__all__ = ["Supervised", "Unsupervised"]


class Supervised(Learner):
    """Generic supervised learning task."""

    # ------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------
    def get_inputs(self, batch: Dict[str, Tensor]) -> Any:
        """Extract inputs from ``batch``."""
        return batch["input"]

    def get_targets(self, batch: Dict[str, Tensor]) -> Any:
        """Extract targets from ``batch``."""
        return batch["target"]

    def forward_batch(self, inputs: Any) -> Any:
        """Forward pass over ``inputs``."""
        return self(inputs)

    def compute_loss(self, outputs: Any, targets: Any) -> Tensor:
        """Compute loss given ``outputs`` and ``targets``."""
        return self.criterion(outputs, targets)


class Unsupervised(Learner):
    """Generic unsupervised learning task."""

    def get_inputs(self, batch: Dict[str, Tensor]) -> Any:
        return batch["input"]

    def forward_batch(self, inputs: Any) -> Any:
        return self(inputs)

    def compute_loss(self, outputs: Any) -> Tensor:
        return self.criterion(outputs)

