"""Utilities for post-processing model outputs.

This module defines the :class:`Predictor` base class used to convert raw
model outputs into task-specific predictions. Predictor objects are
callable and are typically assigned to a :class:`Learner` instance to
customise the behaviour of ``predict_step``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor


class Predictor(ABC):
    """Callable used by :class:`Learner` to post-process model outputs."""

    @abstractmethod
    def __call__(self, outputs: Tensor) -> Any:
        """Convert model outputs into predictions.

        Parameters
        ----------
        outputs : Tensor
            Raw outputs produced by a model's forward pass.

        Returns
        -------
        Any
            Processed predictions ready to be consumed by downstream
            applications.
        """
