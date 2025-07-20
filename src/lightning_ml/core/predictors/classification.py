"""Predictor for classification problems."""

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Union
from collections.abc import Sequence

from torch import Tensor

from ..utils.enums import Registries
from ..utils.registry import register
from ..core import Predictor

__all__ = ["Classification"]


@register(Registries.PREDICTOR)
class Classification(Predictor):
    """Predictor for classification tasks.

    Converts model outputs into class indices or string labels.
    """

    valid_return_types = {"indices", "labels", "probs"}

    def __init__(
        self,
        softmax: bool = True,
        class_list: Sequence[str] | None = None,
        default_return: str = "indices",
    ) -> None:
        """Initialize the Classification predictor.

        Args:
            softmax (bool): Whether to apply softmax to outputs before prediction.
            class_list (Optional[Sequence[str]]): List of labels for each class index.
            default_return (str): Default return type. One of {"indices", "labels", "probs"}.
                "labels" requires `class_list` to be specified.
        """
        self.softmax = softmax
        self.class_list = class_list
        # Validate and store the default return type
        if default_return not in self.valid_return_types:
            raise ValueError(
                f"`default_return` must be one of {self.valid_return_types}, got {default_return!r}"
            )
        if default_return == "labels" and not class_list:
            warnings.warn(
                "`class_list` not provided; falling back to class indices for default_return",
                UserWarning,
            )
            default_return = "indices"
        self.default_return = default_return

    def _map_labels(self, o: int | list[Any]) -> str | list[str]:
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
        self,
        outputs: Tensor,
        *,
        return_type: str | None = None,
    ) -> Tensor | list[str]:
        """Predict class indices, labels, or probabilities from model outputs.

        Args:
            outputs (Tensor): Raw model outputs (logits or probabilities).
            return_type (Optional[str]): Desired output type—one of {"indices", "labels", "probs"}.
                If None, falls back to the default specified when the predictor was constructed
                (`default_return`).

        Returns:
            Union[Tensor, List[str]]: Class indices as a Tensor, probabilities as a Tensor,
            or a nested list of class labels.

        Raises:
            ValueError: If an invalid return_type is provided.
        """
        desired = return_type if return_type is not None else self.default_return

        if desired not in self.valid_return_types:
            raise ValueError(
                f"`return_type` must be one of {self.valid_return_types}, got {desired!r}"
            )

        # Handle label availability
        if desired == "labels" and not self.class_list:
            warnings.warn(
                "`class_list` not provided; returning class indices instead",
                UserWarning,
            )
            desired = "indices"

        if self.softmax:
            outputs = outputs.softmax(dim=-1)
            if desired == "probs":
                return outputs

        class_idx = outputs.argmax(dim=-1)

        if desired == "indices":
            return class_idx

        return self._map_labels(class_idx.tolist())  # desired == "labels"
