"""Abstract base classes for Lightning-ML."""

from .learner import Learner
from .predictor import Predictor
from .school import School

__all__ = ["Learner", "Predictor", "School"]
