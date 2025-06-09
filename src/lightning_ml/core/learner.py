"""Base :class:`~pytorch_lightning.LightningModule` implementations.

This module defines the :class:`Learner` class which acts as a thin
wrapper around a PyTorch ``nn.Module`` and handles training logic,
metrics and prediction post-processing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import pytorch_lightning as pl
from torch import Tensor, nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection

from .predictor import Predictor


class Learner(pl.LightningModule, ABC):
    """Abstract base class for training Lightning models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        *,
        data: Optional[pl.LightningDataModule] = None,
        criterion: Optional[Callable[[Any, Any], Tensor] | nn.Module] = None,
        metrics: Optional[dict[str, MetricCollection]] = None,
        scheduler: Optional[_LRScheduler] = None,
        predictor: Optional[Predictor] = None,
    ) -> None:
        """Initialise the learner.

        Parameters
        ----------
        model : nn.Module
            Neural network model to be trained.
        optimizer : Optimizer
            Optimiser instance to be used during training.
        data : Optional[pytorch_lightning.LightningDataModule], optional
            Datamodule providing the data loaders, by default ``None``.
        criterion : Callable or nn.Module, optional
            Loss function used for optimisation, by default ``None``.
        metrics : dict[str, MetricCollection], optional
            Mapping from stage name (``"train"``, ``"val"``, ``"test"``) to metric
            collections, by default ``None``.
        scheduler : _LRScheduler, optional
            Learning rate scheduler, by default ``None``.
        predictor : Predictor, optional
            Callable used to post-process raw model outputs during prediction,
            by default ``None``.
        """
        super().__init__()
        self.model = model
        self.data = data
        self.criterion = criterion
        self.metrics = metrics or {}
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.predictor = predictor

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def predictor(self) -> Callable[[Tensor], Any]:
        """Prediction post-processing callable."""
        return self._predictor

    @predictor.setter
    def predictor(self, fn: Predictor | Callable[[Tensor], Any] | None) -> None:
        if fn is None:
            fn = lambda x: x  # identity fn
        if not callable(fn):
            raise TypeError("predictor must be callable")
        self._predictor = fn

    # ------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------
    def get_inputs(self, batch: Any) -> Any:
        """Extract inputs from ``batch``.

        The default implementation expects ``batch`` to be a mapping with an
        ``"input"`` key. Subclasses should override this method to customise
        how inputs are retrieved.
        """

        return batch["input"]

    def get_targets(self, batch: Any) -> Any:
        """Extract targets from ``batch``.

        The default implementation looks for a ``"target"`` key and will raise
        ``KeyError`` if it does not exist. Unsupervised tasks can override this
        to simply return ``None``.
        """

        return batch["target"]

    def forward_batch(self, inputs: Any) -> Any:
        """Run a forward pass on ``inputs`` using ``self.model``."""

        return self(inputs)

    def compute_loss(self, outputs: Any, targets: Any | None = None) -> Tensor:
        """Compute the training loss.

        Parameters
        ----------
        outputs : Any
            Model outputs from :meth:`forward_batch`.
        targets : Any, optional
            Targets extracted by :meth:`get_targets`. Can be ``None`` for
            unsupervised tasks.
        """

        if targets is None:
            if self.criterion is None:
                raise RuntimeError("criterion must be provided for loss computation")
            return self.criterion(outputs)
        return self.criterion(outputs, targets)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def step(self, batch: Any) -> Dict[str, Any]:
        """Perform a single optimisation step.

        This method orchestrates a typical training iteration using the
        template methods defined above. Subclasses can override individual
        template methods or the ``step`` method itself for full control.
        """

        inputs = self.get_inputs(batch)
        outputs = self.forward_batch(inputs)
        try:
            targets = self.get_targets(batch)
        except KeyError:
            targets = None
        loss = self.compute_loss(outputs, targets)
        out: Dict[str, Any] = {"output": outputs, "loss": loss}
        if targets is not None:
            out["target"] = targets
        return out

    def forward(
        self, *args: Any, **kwargs: Any
    ) -> Any:  # pragma: no cover - thin wrapper
        return self.model(*args, **kwargs)

    def _shared_step(self, batch: Any, stage: str) -> Tensor:
        out = self.step(batch)
        self.log(
            f"{stage}_loss",
            out["loss"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )

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

    # -------------------- training/validation/test --------------------
    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def on_train_epoch_start(self) -> None:
        if "train" in self.metrics:
            self.metrics["train"].reset()

    def on_validation_epoch_start(self) -> None:
        if "val" in self.metrics:
            self.metrics["val"].reset()

    def on_test_epoch_start(self) -> None:
        if "test" in self.metrics:
            self.metrics["test"].reset()

    # ---------------------- prediction ----------------------
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        out = self.step(batch)
        return self.predictor(out.get("output"))

    # -------------------- optimizers --------------------
    def configure_optimizers(self):  # pragma: no cover - simple passthrough
        if self.scheduler is None:
            return self.optimizer
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
        }
