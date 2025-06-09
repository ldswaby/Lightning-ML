import random
from typing import Any, Sequence

from .abstract import ContastiveDatasetBase
from .labelled import LabelledDataset


class ContrastiveLabelledDataset(LabelledDataset, ContastiveDatasetBase):
    """
    In-memory contrastive dataset for labelled data.

    Each sample returns an anchor and a positive sample with the same label.
    """

    def __init__(
        self,
        inputs: Sequence[Any],
        targets: Sequence[Any],
        *,
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        LabelledDataset.__init__(self, inputs, targets)
        # Build mapping from label to list of indices
        self._label_to_idxs = {}
        for idx, label in enumerate(self._targets):
            self._label_to_idxs.setdefault(label, []).append(idx)

    def get_positive(self, idx: int) -> Any:
        """
        Retrieve a positive sample (same label as anchor).
        Falls back to anchor if no other positive exists.
        """
        label = self._targets[idx]
        pos_idxs = self._label_to_idxs[label]
        if len(pos_idxs) < 2:
            return self._inputs[idx]
        pos_idx = idx
        while pos_idx == idx:
            pos_idx = random.choice(pos_idxs)
        return self._inputs[pos_idx]


class ContrastiveLabelledDataset(LabelledDataset):
    """
    In-memory contrastive dataset for labelled data.

    Each sample returns an anchor and a positive sample with the same label.
    """

    def __init__(self, inputs: Sequence[Any], targets: Sequence[Any]) -> None:
        """
        Initialize the contrastive dataset.

        Args:
            inputs (Sequence[Any]): Input data sequence.
            targets (Sequence[Any]): Target labels corresponding to inputs.
        """
        super().__init__(inputs, targets)
        # Build mapping from label to list of indices
        self._label_to_idxs = {}
        for idx, label in enumerate(self._targets):
            self._label_to_idxs.setdefault(label, []).append(idx)

    def get_input(self, idx: int) -> Any:
        """
        Retrieve the anchor at idx and a random positive sample.

        Args:
            idx (int): Index of the anchor sample.

        Returns:
            Any: Anchor input data.
        """
        anchor = self._inputs[idx]
        label = self._targets[idx]
        pos_idxs = self._label_to_idxs[label]
        if len(pos_idxs) < 2:
            # Only one sample for specified label
            return anchor, anchor

        # Return ranodm sample with same label
        pos_idx = idx
        while pos_idx == idx:
            pos_idx = random.choice(pos_idxs)
        positive = self._inputs[pos_idx]
        return anchor, positive
