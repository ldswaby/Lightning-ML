from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import pytorch_lightning as pl
from torch import Tensor, nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection


class Task(pl.LightningModule, ABC):
    """
    Build all the ML project components
    TODO: have this have a fixed __init__ that builds all componnets from
    cfg (or have them built elsewhere then passed into Task as clasmethod?)

    NOTE: we need child classes to at least have training_step
    """

    def __init__(
        self,
        data: pl.LightningDataModule,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        metrics: Optional[dict[str, MetricCollection]] = None,
        scheduler: Optional[_LRScheduler] = None,
    ) -> None:
        super().__init__()
        self.data = data
        self.model = model
        self.criterion = criterion
        self.metrics = metrics or {}
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        """
        Configure optimisers and, optionally, LR schedulers for
        PyTorch-Lightning.
        """
        # TODO: what about scheduler?
        return self.optimizer

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

    @abstractmethod
    def training_step(self, batch, batch_idx: int) -> Tensor:
        """Task-specific training logic. Must return loss tensor"""
