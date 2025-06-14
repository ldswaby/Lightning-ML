"""Implementations of contrastive learning paradigms."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from torch import Tensor

from .abstract import Supervised, Unsupervised

__all__ = ["ContrastiveSupervised", "ContrastiveUnsupervised"]


class ContrastiveSupervised(Supervised):
    """Supervised contrastive learning task."""

    def get_inputs(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        return batch["view1"], batch["view2"]

    def forward_batch(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        v1, v2 = inputs
        z1 = self(v1)
        z2 = self(v2)
        return z1, z2

    def compute_loss(self, outputs: Tuple[Tensor, Tensor], targets: Tensor) -> Tensor:
        z1, z2 = outputs
        return self.criterion(z1, z2, targets)


class ContrastiveUnsupervised(Unsupervised):
    """Unsupervised contrastive learning task."""

    def get_inputs(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        return batch["view1"], batch["view2"]

    def forward_batch(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        v1, v2 = inputs
        z1 = self(v1)
        z2 = self(v2)
        return z1, z2

    def compute_loss(self, outputs: Tuple[Tensor, Tensor]) -> Tensor:
        z1, z2 = outputs
        return self.criterion(z1, z2)

