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


class PredictorMixin:
    """Mixin that adds a Lightning-compatible predict_step."""

    @abstractmethod
    def post_process(self, outputs):
        """Convert Learner.predict_step outputs into task-specific predictions."""

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Predict step to override Problem.predict_step via MRO

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_
            dataloader_idx (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        output = super().predict_step(batch, batch_idx, dataloader_idx)
        return self.post_process(output)
