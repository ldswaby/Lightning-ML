"""Predictor for regression tasks."""

from __future__ import annotations

from torch import Tensor

from . import PREDICTOR_REG
from ..core import Predictor

__all__ = ["Regression"]


@PREDICTOR_REG.register()
class Regression(Predictor):
    """Return regression outputs as 1D tensors."""

    def __call__(self, outputs: Tensor) -> Tensor:
        return outputs.squeeze(-1)

