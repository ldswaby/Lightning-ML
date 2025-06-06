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

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import pytorch_lightning as pl
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .problem import Problem


class Task(pl.LightningModule, ABC):
    """
    Paradigm-agnostic Lightning template.
    Subclasses must implement `step` to compute a dict that at minimum
    contains a `'loss'` Tensor. Anything else is optional.
    """

    def __init__(self, problem: Problem) -> None:
        """
        Initializes the Task with data, model, loss, optimizer, metrics, and scheduler.

        Args:
            data (pl.LightningDataModule): Data module for providing data loaders.
            model (nn.Module): Neural network model to be trained.
            criterion (nn.Module): Loss function used during training.
            optimizer (Optimizer): Optimizer class or a callable that returns an optimizer instance when given parameters.
            metrics (dict[str, MetricCollection], optional): Dictionary mapping stage names ('train', 'val', 'test') to MetricCollection instances. Defaults to None.
            scheduler (_LRScheduler, optional): Learning rate scheduler instance. Defaults to None.
        """
        super().__init__()
        self._problem = problem

    @abstractmethod
    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        """Takes in a batch of data and returns model outputs

        Returns:
            Any: _description_
        """
        pass

    def __getattr__(self, name: str):
        """Delegates everything besides the above methods to the wrapped
        object
        """
        return getattr(self._problem, name)
