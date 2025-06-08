from __future__ import annotations

from typing import Dict, Tuple

from torch import Tensor

from .abstract import Supervised, Unsupervised

__all__ = ["ContrastiveSupervised", "ContrastiveUnsupervised"]


class ContrastiveSupervised(Supervised):
    """Supervised contrastive learning task.

    Expects batches to contain two augmented views of each input and
    corresponding targets. The criterion should accept ``(z1, z2, target)``
    and return a scalar loss tensor.
    """

    def parse_batch(self, batch: Dict[str, Tensor]) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        return (batch["view1"], batch["view2"]), batch["target"]

    def compute_loss(self, outputs: Tuple[Tensor, Tensor], target: Tensor) -> Tensor:
        z1, z2 = outputs
        return self.criterion(z1, z2, target)

    def step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        (view1, view2), target = self.parse_batch(batch)
        z1 = self(view1)
        z2 = self(view2)
        loss = self.compute_loss((z1, z2), target)
        return {"output": (z1, z2), "target": target, "loss": loss}


class ContrastiveUnsupervised(Unsupervised):
    """Unsupervised contrastive learning task.

    Expects batches containing two augmented views of each sample. The
    criterion should accept ``(z1, z2)`` and return a scalar loss tensor.
    """

    def parse_batch(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        return batch["view1"], batch["view2"]

    def compute_loss(self, outputs: Tuple[Tensor, Tensor]) -> Tensor:
        z1, z2 = outputs
        return self.criterion(z1, z2)

    def step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        view1, view2 = self.parse_batch(batch)
        z1 = self(view1)
        z2 = self(view2)
        loss = self.compute_loss((z1, z2))
        return {"output": (z1, z2), "loss": loss}
