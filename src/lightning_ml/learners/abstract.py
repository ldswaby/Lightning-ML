"""Base learner implementations for common paradigms."""

from __future__ import annotations

from typing import Any, Dict, Optional

from torch import Tensor

from ..core import Learner

__all__ = ["Supervised", "Unsupervised"]


class Supervised(Learner):
    """Generic supervised learning task."""

    def process_batch(self, batch: Dict[str, Any]) -> Any:
        """Extract inputs from ``batch`` for `self.forward`."""
        return batch["input"]

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward hook to underlying model `self.model`"""
        return self.model(*args, **kwargs)

    def compute_loss(self, model_outputs: Any, targets: Optional[Any] = None) -> Tensor:
        """Compute loss given raw ``model_outputs`` and ``targets``."""
        return self.criterion(model_outputs, targets)


class Unsupervised(Learner):
    """Generic unsupervised learning task."""

    def process_batch(self, batch: Dict[str, Any]) -> Any:
        """Extract inputs from ``batch`` for model."""
        return batch["input"]

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward hook to underlying model `self.model`"""
        return self.model(*args, **kwargs)

    def compute_loss(self, model_outputs: Any, targets: Optional[Any] = None) -> Tensor:
        """Compute loss given raw ``model_outputs`` and ``targets``."""
        return self.criterion(model_outputs)
