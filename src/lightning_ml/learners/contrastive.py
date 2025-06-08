from __future__ import annotations

from typing import Dict

from torch import Tensor

from .abstract import Supervised, Unsupervised

__all__ = ["ContrastiveSupervised", "ContrastiveUnsupervised"]


class ContrastiveSupervised(Supervised):
    """Supervised contrastive learning task.

    Expects batches to contain two augmented views of each input and
    corresponding targets. The criterion should accept ``(z1, z2, target)``
    and return a scalar loss tensor.
    """

    def step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}
        z1 = self(batch["view1"])
        z2 = self(batch["view2"])
        target = batch["target"]
        out["output"] = (z1, z2)
        out["target"] = target
        out["loss"] = self.criterion(z1, z2, target)
        return out


class ContrastiveUnsupervised(Unsupervised):
    """Unsupervised contrastive learning task.

    Expects batches containing two augmented views of each sample. The
    criterion should accept ``(z1, z2)`` and return a scalar loss tensor.
    """

    def step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}
        z1 = self(batch["view1"])
        z2 = self(batch["view2"])
        out["output"] = (z1, z2)
        out["loss"] = self.criterion(z1, z2)
        return out
