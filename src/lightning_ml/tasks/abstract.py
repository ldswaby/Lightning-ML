from typing import Dict

from torch import Tensor

from ..core import Task

__all__ = ["Supervised", "Unsupervised"]


class Supervised(Task):
    """
    Supervised learning base task.

    Assumes that:
    * Batches can be decomposed into (inputs, targets)
    * Parsed loss fn works directly on model outputs with no prior processing
    """

    def step(self, batch) -> Dict[str, Tensor]:
        """_summary_

        Args:
            batch (_type_): _description_

        Returns:
            Dict[str, Tensor]: _description_
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return {
            "loss": loss,
            "preds": logits,
            "targets": y,
        }


class Unsupervised(Task):
    """
    Unsupervised learning base task.

    Assumes that:
    * Batches are just inputs; no targets or metrics.
    * Parsed loss fn works directly on model outputs with no prior processing
    """

    def step(self, batch) -> Dict[str, Tensor]:
        logits = self(batch)
        loss = self.criterion(logits)
        return {"loss": loss}
