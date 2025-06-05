from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torchmetrics import MetricCollection


class BaseLitModule(pl.LightningModule):
    """Basic LightningModule using a model, loss function, optimizer and metrics."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer_cfg: dict[str, Any],
        scheduler_cfg: dict[str, Any] | None,
        metrics: MetricCollection,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        optimizer_cls = getattr(torch.optim, self.optimizer_cfg["name"])
        optimizer = optimizer_cls(self.parameters(), **self.optimizer_cfg.get("params", {}))
        if self.scheduler_cfg:
            scheduler_cls = getattr(torch.optim.lr_scheduler, self.scheduler_cfg["name"])
            scheduler = scheduler_cls(optimizer, **self.scheduler_cfg.get("params", {}))
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.scheduler_cfg.get("monitor"),
                },
            }
        return optimizer

    def training_step(self, batch, batch_idx: int):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.train_metrics.update(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log_dict(self.train_metrics, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.val_metrics.update(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.val_metrics, prog_bar=True)

    def on_validation_epoch_end(self):
        self.train_metrics.reset()
        self.val_metrics.reset()
