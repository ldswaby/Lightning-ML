"""Lightning-ML core package."""

from . import core, data, learners, models, predictors, utils
# Re-export ``lightning_ml.data.datasets`` as ``lightning_ml.datasets`` for
# convenience and to match the public API expected by the tests.
from .data import datasets as datasets

__all__ = ["core", "data", "learners", "models", "predictors", "utils", "datasets"]
