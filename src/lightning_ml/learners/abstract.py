from typing import Any, Dict

from torch import Tensor

from ..core import Learner

__all__ = ["Supervised", "Unsupervised"]


class Supervised(Learner):
    """Generic supervised learning task."""

    # ------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------
    def get_inputs(self, batch: Dict[str, Tensor]) -> Any:
        """Extract model inputs from a batch."""

        return batch["input"]

    def get_targets(self, batch: Dict[str, Tensor]) -> Any:
        """Extract targets from a batch."""

        return batch["target"]

    def forward_batch(self, inputs: Any) -> Any:
        """Forward pass that can be overridden by subclasses."""

        return self(inputs)

    def compute_loss(self, outputs: Any, targets: Any) -> Tensor:
        """Compute the training loss."""

        return self.criterion(outputs, targets)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        inputs = self.get_inputs(batch)
        targets = self.get_targets(batch)
        outputs = self.forward_batch(inputs)
        loss = self.compute_loss(outputs, targets)
        return {"output": outputs, "target": targets, "loss": loss}


class Unsupervised(Learner):
    """Generic unsupervised learning task."""

    def get_inputs(self, batch: Dict[str, Tensor]) -> Any:
        """Extract model inputs from a batch."""

        return batch["input"]

    def forward_batch(self, inputs: Any) -> Any:
        """Forward pass that can be overridden by subclasses."""

        return self(inputs)

    def compute_loss(self, outputs: Any) -> Tensor:
        """Compute the training loss."""

        return self.criterion(outputs)

    def step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        inputs = self.get_inputs(batch)
        outputs = self.forward_batch(inputs)
        loss = self.compute_loss(outputs)
        return {"output": outputs, "loss": loss}
