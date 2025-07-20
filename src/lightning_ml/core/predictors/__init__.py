"""Predictor registry and built-in predictors."""

from __future__ import annotations

from ..utils.registry import Registry

PREDICTOR_REG = Registry("Predictor")

from .classification import *  # noqa: F401,F403
from .regression import *  # noqa: F401,F403

__all__ = ["PREDICTOR_REG", "Classification", "Regression"]

