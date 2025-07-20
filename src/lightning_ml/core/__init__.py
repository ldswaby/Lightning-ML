"""Core abstractions for Lightning-ML."""

from __future__ import annotations

# from .data.datamodule import DataModule
from . import data
from .learner import Learner
from .predictor import Predictor
from .school import School

__all__ = ["Learner", "Predictor", "School", "BaseDataset", "DataModule"]
