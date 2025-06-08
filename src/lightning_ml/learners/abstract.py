from typing import Dict

from torch import Tensor

from ..core import Learner

__all__ = ["Supervised", "Unsupervised"]


class Supervised(Learner):
    """
    Supervised learning base task.

    Assumes that:
    * Batches can be decomposed into (inputs, targets)
    * Parsed loss fn works directly on model outputs with no prior processing
    """

    def step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        TODO: Note that assumed data batches always output dicts with at least
        keys ['inputs', 'targets']. this can eb for a Supervised Dataset object

        Args:
            batch (_type_): _description_

        Returns:
            Dict[str, Tensor]: _description_
        """
        out = {}
        out["targets"] = batch["targets"]
        out["outputs"] = self(batch["inputs"])
        out["loss"] = self.criterion(out["outputs"], batch["targets"])
        return out


class Unsupervised(Learner):
    """
    Unsupervised learning base task.

    Assumes that:
    * Batches are just inputs; no targets or metrics.
    * Parsed loss fn works directly on model outputs with no prior processing
    """

    def step(self, batch) -> Dict[str, Tensor]:
        out = {}
        out["outputs"] = self.predict_step(batch)
        out["loss"] = self.criterion(out["outputs"])
        return out
