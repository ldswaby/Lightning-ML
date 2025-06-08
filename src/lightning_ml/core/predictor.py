"""
Task abstraction for training, validation, and testing of PyTorch Lightning models.

This module defines the Task base class that encapsulates the core components
of a machine learning pipeline, including data modules, models, loss functions,
optimizers, metrics, and schedulers.


# TODO automcaticalay runs self[batch]

Problem can be class below
Task can wrap it with specialized prediction logic in predict_step
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict

from torch import Tensor


class Predictor:
    """Mixin that adds a Lightning-compatible predict_step."""

    @abstractmethod
    def __call__(self, outputs: Dict[str, Tensor]) -> Any:
        """Convert Learner.predict_step outputs into task-specific predictions."""
