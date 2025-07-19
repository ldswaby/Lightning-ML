"""Semi-supervised learning paradigm."""

from __future__ import annotations

from typing import Any, Dict, Optional
from collections.abc import Callable

from torch import Tensor

from . import LEARNER_REG
from ..core import Learner

__all__ = ["SemiSupervised"]


@LEARNER_REG.register()
class SemiSupervised(Learner):
    """Generic semi-supervised learning task.

    Combines supervised and unsupervised objectives using ``criterion`` for
    supervised loss and ``unsupervised_criterion`` for unsupervised loss.
    The two losses are weighted by ``lambda_u``.
    """

    def __init__(
        self,
        *args: Any,
        unsupervised_criterion: Callable[[Any], Tensor],
        lambda_u: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.unsupervised_criterion = unsupervised_criterion
        self.lambda_u = lambda_u

    def batch_forward(self, batch: dict[str, Any]) -> Any:
        """Forward hook to underlying model ``self.model``"""
        return self.model(batch["input"])

    def compute_loss(self, model_outputs: Any, targets: Any | None = None) -> Tensor:
        """Compute weighted semi-supervised loss."""
        loss_u = self.unsupervised_criterion(model_outputs)
        loss_s = self.criterion(model_outputs, targets) if targets is not None else 0
        return loss_s + self.lambda_u * loss_u
