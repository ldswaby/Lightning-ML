"""Predictor registry and built-in predictors."""

from __future__ import annotations

from ..utils.enums import Registries
from ..utils.registry import get_registry

PREDICTOR_REG = get_registry(Registries.PREDICTOR)

from .classification import *  # noqa: F401,F403
from .regression import *  # noqa: F401,F403

__all__ = ["PREDICTOR_REG", "Classification", "Regression"]

