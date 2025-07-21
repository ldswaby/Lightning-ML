"""Abstract base class for Lightning modules.

Learner acts as a thin wrapper around a PyTorch nn.Module and handles
training logic, metric computation, and prediction post-processing.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from collections.abc import Callable

import pytorch_lightning as pl
from torch import Tensor, nn
from torch.nn import ModuleDict
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection

from .predictor import Predictor


class Learner(pl.LightningModule, ABC):
    """Abstract base class for training Lightning models.

    Learner wraps a PyTorch nn.Module to streamline training steps,
    metric logging, and post-processing of predictions.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        *,
        data_module: pl.LightningDataModule | None = None,
        criterion: Callable[[Any, Any], Tensor] | nn.Module | None = None,
        metrics: dict[str, MetricCollection] | None = None,
        scheduler: _LRScheduler | None = None,
        predictor: Predictor | Callable[[Tensor], Any] | None = None,
    ) -> None:
        """Initialize the Learner.

        Args:
            model (nn.Module): Neural network to train.
            optimizer (Optimizer): Optimizer for training.
            data_module (Optional[pl.LightningDataModule], optional): Data module for data loaders. Defaults to None.
            criterion (Optional[Callable[[Any, Any], Tensor] or nn.Module], optional): Loss function. Defaults to None.
        metrics (Optional[Dict[str, MetricCollection]], optional): Mapping of stage name
                to a :class:`~torchmetrics.MetricCollection`. These will be registered
                as modules to ensure they follow the Learner's device.
            scheduler (_LRScheduler, optional): Learning rate scheduler. Defaults to None.
            predictor (Predictor, optional): Post-processing function for predictions. Defaults to None.
        """
        # Initialize the LightningModule base
        super().__init__()
        self.model = model
        self.data_module = data_module
        self.criterion = criterion
        # register metrics so Lightning moves them to the correct device
        self.metrics = ModuleDict(metrics or {})
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.predictor = predictor  # fallback to identity fn

    @abstractmethod
    def batch_forward(self, batch: dict[str, Any]) -> Any:
        """Extract inputs from ``batch`` and runs through model.

        `Outputs for this function go directly in self.compute_loss`
        """

    @abstractmethod
    def compute_loss(self, model_outputs: Any, targets: Any | None = None) -> Tensor:
        """Compute loss from model outputs and targets.

        Args:
            model_outputs (Any): Outputs from the model.
            targets (Optional[Any], optional): Ground truth targets. Defaults to None.

        Returns:
            Tensor: Computed loss value.
        """

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

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward hook"""
        return self.model(*args, **kwargs)

    def step(self, batch: Any, stage: str) -> dict[str, Tensor]:
        """Run a single step for a given stage.

        Args:
            batch (Any): Batch data from DataLoader, typically including inputs and targets.
            stage (str): Stage name, one of "train", "val", or "test".

        Returns:
            Dict[str, Tensor]: Dictionary containing "inputs", "outputs", and "loss" tensors.
        """
        # Prepare inputs and perform forward pass
        logs = {}
        logs["outputs"] = self.batch_forward(batch)
        logs["loss"] = self.compute_loss(logs["outputs"], batch.get("target"))

        # Log the loss for this stage
        self.log(
            f"{stage}_loss",
            logs["loss"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )

        # Compute and log metrics if available
        if stage in self.metrics:
            outputs = logs.get("outputs")
            targets = batch.get("target")
            if outputs is not None and targets is not None:
                metric_vals = self.metrics[stage](outputs, targets)
                self.log_dict(
                    {f"{stage}_{k}": v for k, v in metric_vals.items()},
                    prog_bar=True,
                    on_epoch=True,
                    batch_size=len(batch),
                )

        return logs

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Perform a single training step.

        Args:
            batch (Any): Batch data from DataLoader.
            batch_idx (int): Index of the current batch.

        Returns:
            Tensor: Loss tensor for the training step.
        """
        logs = self.step(batch, "train")
        return logs["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Perform a single validation step.

        Args:
            batch (Any): Batch data from DataLoader.
            batch_idx (int): Index of the current batch.
        """
        self.step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """Perform a single test step.

        Args:
            batch (Any): Batch data from DataLoader.
            batch_idx (int): Index of the current batch.
        """
        self.step(batch, "test")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Perform a prediction step with post-processing.

        Args:
            batch (Any): Batch data from DataLoader.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.

        Returns:
            Any: Post-processed model outputs.
        """
        inputs = self.process_batch(batch)
        outputs = self(*inputs)
        return self.predictor(outputs)

    def on_train_epoch_start(self) -> None:
        """Reset training metrics at epoch start"""
        if "train" in self.metrics:
            self.metrics["train"].reset()

    def on_validation_epoch_start(self) -> None:
        """Reset validation metrics at epoch start"""
        if "val" in self.metrics:
            self.metrics["val"].reset()

    def on_test_epoch_start(self) -> None:
        """Reset test metrics at epoch start"""
        if "test" in self.metrics:
            self.metrics["test"].reset()

    def configure_optimizers(self):
        """Simple passthrough for optimizer (and scheduler if provided)

        Returns:
            Union[Optimizer, Dict[str, Any]]: Single optimizer or dict including scheduler.
        """
        if self.scheduler is None:
            return self.optimizer
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
        }
