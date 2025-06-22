"""Base learner implementations for common paradigms."""

from typing import Any, Dict, Optional

from torch import Tensor

from . import LEARNER_REG
from ..core import Learner


@LEARNER_REG.register()
class Supervised(Learner):
    """Generic supervised learning task."""

    def batch_forward(self, batch: Dict[str, Any]) -> Any:
        """Forward hook to underlying model `self.model`"""
        return self.model(batch["input"])

    def compute_loss(self, model_outputs: Any, targets: Optional[Any] = None) -> Tensor:
        """Compute loss given raw ``model_outputs`` and ``targets``."""
        return self.criterion(model_outputs, targets)
