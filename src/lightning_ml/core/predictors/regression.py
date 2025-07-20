"""Predictor for regression tasks."""

from __future__ import annotations

from torch import Tensor

from ..utils.enums import Registries
from ..utils.registry import register
from ..abstract import Predictor

__all__ = ["Regression"]


@register(Registries.PREDICTOR)
class Regression(Predictor):
    """Return regression outputs as 1D tensors."""

    def __call__(self, outputs: Tensor) -> Tensor:
        return outputs.squeeze(-1)

