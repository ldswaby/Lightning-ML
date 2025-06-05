from __future__ import annotations

from typing import Any, Optional

import pytorch_lightning as pl
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection


class Task(pl.LightningModule):
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
        metrics: MetricCollection,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
    ) -> None:
        super().__init__()
        self.data = data
        self.model = model
        self.criterion = criterion
        self.metrics = metrics
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
