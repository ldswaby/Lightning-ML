"""Wrapper classes to adapt torchvision datasets to Lightning-ML's Dataset API."""

from __future__ import annotations

from typing import Any, Type

from .. import DATASET_REG
from ..abstract import LabelledDatasetBase

__all__ = ["TorchvisionDataset"]

try:
    from torch.utils.data import Dataset as TorchDataset
except Exception as e:  # pragma: no cover - import dependency
    raise ImportError("PyTorch is required for TorchvisionDataset") from e


@DATASET_REG.register()
class TorchvisionDataset(LabelledDatasetBase):
    """
    Wraps a torchvision dataset to adapt it to Lightning-ML's Dataset API.

    This wrapper returns (input, target) tuples for each sample.
    """

    dataset: TorchDataset

    def __init__(
        self, dataset: TorchDataset | Type[TorchDataset], **kwargs: Any
    ) -> None:
        """Initializes the TorchvisionDataset.

        Args:
            dataset (Union[TorchDataset, Type[TorchDataset]]): Either an instantiated torchvision Dataset or the class to instantiate.
            **kwargs: Additional arguments passed to the dataset constructor.
        """
        if isinstance(dataset, type):
            dataset = dataset(**kwargs)
        self.dataset = dataset

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.dataset)

    def _get_tuple(self, idx: int) -> tuple[Any, Any]:
        """Retrieves the (input, target) tuple from the dataset at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[Any, Any]: A tuple containing the input and target.

        Raises:
            ValueError: If the sample is not a tuple of length >= 2.
        """
        sample = self.dataset[idx]
        if not isinstance(sample, tuple) or len(sample) < 2:
            raise ValueError("Underlying dataset must return a tuple (input, target)")
        return sample[0], sample[1]

    def get_input(self, idx: int) -> Any:
        """Retrieves only the input from the sample at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Any: The input component of the sample.
        """
        inp, _ = self._get_tuple(idx)
        return inp

    def get_target(self, idx: int) -> Any:
        """Retrieves only the target from the sample at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Any: The target component of the sample.
        """
        _, tgt = self._get_tuple(idx)
        return tgt
