"""Lightning-ML core package."""

from . import core
from .core import data

# Re-export ``lightning_ml.core.data.datasets`` as ``lightning_ml.datasets``
from .core.data import datasets as datasets

__all__ = ["core", "data", "datasets"]
