"""Implementations of contrastive learning paradigms."""

from typing import Any, Dict, Optional

from torch import Tensor

from .supervised import Supervised
from .unsupervised import Unsupervised

__all__ = ["ContrastiveSupervised", "ContrastiveUnsupervised"]


class ContrastiveSupervised(Supervised):
    """Supervised contrastive learning task."""

    def batch_forward(self, batch: Dict[str, Any]) -> Any:
        """Forward hook to underlying model `self.model`"""
        z1 = self.model(batch["input"])
        z2 = self.model(batch["positive"])
        return z1, z2

    def compute_loss(self, model_outputs: Any, targets: Optional[Any] = None) -> Tensor:
        """Compute loss given raw ``model_outputs`` and ``targets``."""
        return self.criterion(*model_outputs, targets)


class ContrastiveUnsupervised(Unsupervised):
    """Unsupervised contrastive learning task."""

    def batch_forward(self, batch: Dict[str, Any]) -> Any:
        """Forward hook to underlying model `self.model`"""
        z1 = self.model(batch["input"])
        z2 = self.model(batch["positive"])
        return z1, z2

    def compute_loss(self, model_outputs: Any, targets: Optional[Any] = None) -> Tensor:
        """Compute loss given raw ``model_outputs`` and ``targets``."""
        return self.criterion(*model_outputs)
