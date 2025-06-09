"""Core abstractions for Lightning-ML."""

from __future__ import annotations

from .dataset import BaseDataset
from .learner import Learner
from .predictor import Predictor
from .project import Project

__all__ = ["Learner", "Predictor", "Project", "BaseDataset"]
