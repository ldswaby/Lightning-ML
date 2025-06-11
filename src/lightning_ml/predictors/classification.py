"""Predictor for classification problems."""

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Sequence, Union

from torch import Tensor

from ..core import Predictor

__all__ = ["Classification"]


class Classification(Predictor):
    """Predictor for classification tasks.

    Converts model outputs into class indices or string labels.
    """

    def __init__(
        self,
        softmax: bool = True,
        class_list: Optional[Sequence[str]] = None,
        return_labels: bool = False,
    ) -> None:
        """Initialize the Classification predictor.

        Args:
            softmax (bool): Whether to apply softmax to outputs before prediction.
            class_list (Optional[Sequence[str]]): List of labels for each class index.
            return_labels (bool): If True, __call__ returns labels instead of indices by default.
        """
        self.softmax = softmax
        self.class_list = class_list
        self.return_labels = return_labels
        if self.return_labels and not class_list:
            warnings.warn(
                "`class_list` not provided; returning class indices instead",
                UserWarning,
            )
            self.return_labels = False

    def _map_labels(self, o: Union[int, List[Any]]) -> Union[str, List[str]]:
        """Recursively map class indices to labels.

        Args:
            o (Union[int, List[Any]]): A class index or nested list of indices.

        Returns:
            Union[str, List[str]]: Corresponding label or nested list of labels.
        """
        if isinstance(o, list):
            return [self._map_labels(i) for i in o]
        return self.class_list[o]

    def __call__(
        self, outputs: Tensor, *, return_labels: Optional[bool] = None
    ) -> Union[Tensor, List[str]]:
        """Predict class indices or labels from model outputs.

        Args:
            outputs (Tensor): Raw model outputs (logits or probabilities).
            return_labels (Optional[bool]): Whether to return labels (True) or indices (False).
                If None, uses the default set at initialization.

        Returns:
            Union[Tensor, List[str]]: Class indices as a Tensor if labels are not requested,
            or a nested list of class labels.

        Raises:
            ValueError: If labels are requested but no class_list was provided.
        """
        # Determine whether to return class indices or labels
        if return_labels is None:
            return_labels = self.return_labels
        elif return_labels and not self.class_list:
            warnings.warn(
                "`class_list` not provided; returning class indices instead",
                UserWarning,
            )
            return_labels = False

        if self.softmax:
            outputs = outputs.softmax(dim=-1)

        class_idx = outputs.argmax(dim=-1)
        if not return_labels:  # Return class indices
            return class_idx

        return self._map_labels(class_idx.tolist())
