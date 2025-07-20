"""Core abstractions for Lightning-ML."""

from __future__ import annotations

# from .data.datamodule import DataModule
from . import data
from .abstract.learner import Learner
from .abstract.predictor import Predictor
from .abstract.school import School

__all__ = ["Learner", "Predictor", "School"]
