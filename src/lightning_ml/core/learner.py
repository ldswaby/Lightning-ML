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
from typing import Any, Callable, Dict, Optional, Type

import pytorch_lightning as pl
from torch import Tensor, nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection

from ..utils import bind_classes
from .predictor import Predictor


class Learner(pl.LightningModule, ABC):
    """
    Paradigm-agnostic Lightning template.
    Subclasses must implement `step` to compute a dict that at minimum
    contains a `'loss'` Tensor. Anything else is optional.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        data: Optional[pl.LightningDataModule] = None,
        criterion: Optional[Callable | nn.Module] = None,
        metrics: Optional[dict[str, MetricCollection]] = None,
        scheduler: Optional[_LRScheduler] = None,
        predictor: Optional[Predictor] = None,
    ) -> None:
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
        # self.save_hyperparameters(ignore=["model", "data", "criterion", "metrics"])
        self.model = model
        self.data = data
        self.criterion = criterion  # may be None for e.g. GANs, BYOL, …
        self.metrics = metrics or {}
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.predictor = predictor  # fallback to identity fn

    @property
    def predictor(self) -> Callable:
        """Callable that post-processes the output of :meth:`predict_step`.

        You can re-assign it after instantiation:

        ```python
        learner.predictor = my_custom_predictor
        ```
        """
        return self._predictor

    @predictor.setter
    def predictor(self, fn: Callable | None) -> None:
        """Validate and (re)assign the predictor.

        If *fn* is ``None`` we fall back to an identity function so
        :pymeth:`predict_step` continues to work.
        """
        if fn is None:
            fn = lambda x: x  # identity
        if not callable(fn):
            raise TypeError("predictor must be callable")
        self._predictor = fn

    @abstractmethod
    def step(self, batch: Any) -> Dict[str, Any]:
        """
        Must use self.predict_step if it is is to be compatible with Task wrappers

        Must return at least {'loss': Tensor}. Can also return
        'output', 'target', additional tensors or scalars for logging.
        """

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    # shared wrapper used by train/val/test hooks
    def _shared_step(self, batch: Any, stage: str) -> Tensor:
        """Logic shared by train, test, and validation steps.

        Args:
            batch (Any): _description_
            stage (str): _description_

        Returns:
            Tensor: _description_
        """
        out = self.step(batch)  # user-defined
        self.log(
            f"{stage}_loss",
            out["loss"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )

        # Optional metrics
        if stage in self.metrics:
            outputs = out.get("output")
            targets = out.get("target")
            if outputs is not None and targets is not None:
                metric_vals = self.metrics[stage](outputs, targets)
                self.log_dict(
                    {f"{stage}_{k}": v for k, v in metric_vals.items()},
                    prog_bar=True,
                    on_epoch=True,
                    batch_size=len(batch),
                )
        return out["loss"]

    def training_step(self, batch, batch_idx) -> Tensor:
        """Training step logic

        Returns:
            Tensor: loss
        """
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx) -> None:
        """Validation step logic"""
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx) -> None:
        """Test step logic"""
        self._shared_step(batch, "test")

    def on_train_epoch_start(self) -> None:
        """Resets train metrics per epoch"""
        if "train" in self.metrics:
            self.metrics["train"].reset()

    def on_validation_epoch_start(self) -> None:
        """Resets val metrics per epoch"""
        if "val" in self.metrics:
            self.metrics["val"].reset()

    def on_test_epoch_start(self) -> None:
        """Resets test metrics per epoch"""
        if "test" in self.metrics:
            self.metrics["test"].reset()

    def configure_optimizers(self) -> Optimizer:
        """
        Configure optimisers and, optionally, LR schedulers for
        PyTorch-Lightning.
        """
        # TODO: what about scheduler?
        return self.optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Predict step to override Problem.predict_step via MRO

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_
            dataloader_idx (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        out = self.step(batch)  # user-defined
        return self.predictor(out["output"])

    # @classmethod
    # def with_predictor(cls, predictor: Type[PredictorMixin], *args, **kwargs):
    #     """Combines a predictor mixin with a learner, for comprehesnive inference"""
    #     _cls = bind_classes(cls, predictor)
    #     return _cls(*args, **kwargs)

    ##########
