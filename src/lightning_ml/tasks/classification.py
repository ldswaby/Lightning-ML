"""Classification task built on top of the generic :class:`Task`."""

from __future__ import annotations

from torch import Tensor

from ..core import Task

__all__ = ["Classification"]


class Classification(Task):
    """Basic classification task."""

    def _shared_step(self, batch: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        return logits, y, loss

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        logits, targets, loss = self._shared_step(batch)
        self.compute_metrics(logits, targets, split="train", log=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        logits, targets, loss = self._shared_step(batch)
        self.compute_metrics(logits, targets, split="val", log=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        logits, targets, loss = self._shared_step(batch)
        self.compute_metrics(logits, targets, split="test", log=True)
        self.log("test", loss, prog_bar=True)
        return loss
