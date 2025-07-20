"""
Contrastive dataset implementations.

This module provides:
- ContrastiveLabelledDataset: in-memory contrastive dataset with labels.
- ContrastiveUnlabelledDataset: in-memory contrastive dataset without labels.
- TripletLabelledDataset: in-memory triplet dataset using class labels.
"""

import random
from typing import Any, Optional
from collections.abc import Callable, Sequence

from . import DATASET_REG
from .abstract import ContastiveDatasetBase, TripletDatasetBase
from .labelled import LabelledDataset
from .unlabelled import UnlabelledDataset

__all__ = [
    "ContrastiveLabelledDataset",
    "ContrastiveUnlabelledDataset",
    "TripletDataset",
]


@DATASET_REG.register()
class ContrastiveLabelledDataset(LabelledDataset, ContastiveDatasetBase):
    """
    In-memory contrastive dataset for labelled data.

    Each sample returns an anchor and a positive sample with the same label.

    Attributes:
        _label_to_idxs (Dict[Any, List[int]]): Mapping from label to list of sample indices.
    """

    def __init__(
        self,
        inputs: Sequence[Any],
        targets: Sequence[Any],
        *,
        transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[Any], Any] | None = None,
    ) -> None:
        """
        Initialize the contrastive-labelled dataset.

        Args:
            inputs (Sequence[Any]): Input data sequence.
            targets (Sequence[Any]): Target labels corresponding to inputs.
            transform (Optional[Callable[[Any], Any]]): Transformation applied to inputs.
            target_transform (Optional[Callable[[Any], Any]]): Transformation applied to targets.
        """
        super().__init__(
            inputs, targets, transform=transform, target_transform=target_transform
        )
        # Build mapping from label to list of indices
        self._label_to_idxs = {}
        for idx, label in enumerate(self._targets):
            self._label_to_idxs.setdefault(label, []).append(idx)

    def get_positive(self, idx: int) -> Any:
        """
        Retrieve a positive sample (same label as anchor).

        Args:
            idx (int): Index of the anchor sample.

        Returns:
            Any: Positive sample data corresponding to a different sample of the same label.
        """
        label = self.get_target(idx)
        pos_idxs = self._label_to_idxs[label]
        if len(pos_idxs) < 2:
            return self.get_input(idx)
        pos_idx = idx
        while pos_idx == idx:
            pos_idx = random.choice(pos_idxs)
        return self.get_input(pos_idx)


@DATASET_REG.register()
class ContrastiveUnlabelledDataset(UnlabelledDataset, ContastiveDatasetBase):
    """
    Contrastive dataset without labels.

    Each sample returns two stochastically augmented views of the same input.

    Args:
        inputs (Sequence[Any]): Raw input sequence.
        transform (Optional[Callable[[Any], Any]]): Augmentation callable applied independently to both views.
    """

    def __init__(
        self,
        inputs: Sequence[Any],
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        """
        Initialize the unlabelled contrastive dataset.

        Args:
            inputs (Sequence[Any]): Raw input sequence.
            transform (Optional[Callable[[Any], Any]]): Augmentation callable applied independently to both views.
        """
        super().__init__(inputs, transform=transform)

    def get_positive(self, idx: int) -> Any:
        """
        Return an independent augmented view of the same sample.

        Args:
            idx (int): Index of the input sample.

        Returns:
            Any: Augmented positive view of the sample.
        """
        return self.get_input(idx)


@DATASET_REG.register()
class TripletDataset(ContrastiveLabelledDataset, TripletDatasetBase):
    """
    Triplet dataset returning (anchor, positive, negative) given class labels.

    Attributes:
        _label_to_idxs (Dict[Any, List[int]]): Mapping from label to list of sample indices.
    """

    def get_negative(self, idx: int) -> Any:
        """
        Retrieve a negative sample (different label than anchor).

        Args:
            idx (int): Index of the anchor sample.

        Returns:
            Any: Negative sample data from a different label class.
        """
        anchor_label = self.get_target(idx)
        neg_label = anchor_label
        while neg_label == anchor_label:
            neg_label = random.choice(list(self._label_to_idxs.keys()))
        neg_idx = random.choice(self._label_to_idxs[neg_label])
        return self.get_input(neg_idx)
