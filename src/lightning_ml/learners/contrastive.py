"""Implementations of contrastive learning paradigms."""

from typing import Any, Dict, Optional

from torch import Tensor

from ..core import Learner

__all__ = ["Contrastive"]


class Contrastive(Learner):
    """Contrastive learning task. Supports both supervised and unsupervised."""

    def batch_forward(self, batch: Dict[str, Any]) -> Any:
        """Forward hook to underlying model `self.model`"""
        z1 = self.model(batch["input"])
        z2 = self.model(batch["positive"])
        return z1, z2

    def compute_loss(self, model_outputs: Any, targets: Optional[Any] = None) -> Tensor:
        """Compute loss given raw ``model_outputs`` and ``targets``.

        Assumes loss fn takes input_embedding, positive_emedding, (label)

        """
        return self.criterion(
            *model_outputs, *([targets] if targets is not None else [])
        )
