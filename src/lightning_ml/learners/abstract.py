from typing import Dict, Tuple

from torch import Tensor

from ..core import Learner

__all__ = ["Supervised", "Unsupervised"]


class Supervised(Learner):
    """Base class for supervised learning tasks."""

    def parse_batch(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        return batch["input"], batch["target"]

    def compute_loss(self, outputs: Tensor, target: Tensor) -> Tensor:
        return self.criterion(outputs, target)

    def step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        inputs, target = self.parse_batch(batch)
        outputs = self(inputs)
        loss = self.compute_loss(outputs, target)
        return {"output": outputs, "target": target, "loss": loss}


class Unsupervised(Learner):
    """
    Unsupervised learning base task.

    Assumes that:
    * Batches are just inputs; no targets or metrics.
    * Parsed loss fn works directly on model outputs with no prior processing
    """

    def parse_batch(self, batch: Dict[str, Tensor]) -> Tensor:
        return batch["input"]

    def compute_loss(self, outputs: Tensor) -> Tensor:
        return self.criterion(outputs)

    def step(self, batch) -> Dict[str, Tensor]:
        inputs = self.parse_batch(batch)
        outputs = self(inputs)
        loss = self.compute_loss(outputs)
        return {"output": outputs, "loss": loss}
