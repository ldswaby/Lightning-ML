"""Base learner implementations for common paradigms."""

from typing import Any, Dict, Optional

from torch import Tensor

from ..utils.enums import Registries
from ..utils.registry import register
from ..abstract import Learner


@register(Registries.LEARNER)
class Supervised(Learner):
    """Generic supervised learning task."""

    def batch_forward(self, batch: dict[str, Any]) -> Any:
        """Forward hook to underlying model `self.model`"""
        return self.model(batch["input"])

    def compute_loss(self, model_outputs: Any, targets: Any | None = None) -> Tensor:
        """Compute loss given raw ``model_outputs`` and ``targets``."""
        return self.criterion(model_outputs, targets)
