"""
Utilities for splitting PyTorch Datasets using scikit-learn cross-validators.

This module provides a function to convert sklearn cross-validation splits
into torch.utils.data.Subset objects for train and validation.
"""

from typing import Dict, List

from sklearn.model_selection._split import BaseCrossValidator
from torch.utils.data import Subset

from ..core import BaseDataset


def validation_split(
    dataset: BaseDataset, cv: BaseCrossValidator
) -> List[Dict[str, Subset]]:
    """Split a PyTorch Dataset into train and validation subsets.

    TODO: make support groups?

    Args:
        dataset: A torch.utils.data.Dataset instance supporting __len__ and
            __getitem__. Each subset is indexed via dataset[indices].
        cv: An instantiated scikit-learn cross-validator.

    Returns:
        A list of dictionaries, each with keys "train" and "val"
        mapped to torch.utils.data.Subset instances corresponding to the
        train and validation splits for each fold.
    """
    n_samples = len(dataset)
    y = getattr(dataset, "targets", None)  # for LabelledDatasetBase subclasses
    splits: List[Dict[str, Subset]] = []

    for train_idx, val_idx in cv.split(range(n_samples), y):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        splits.append({"train": train_subset, "val": val_subset})

    return splits
