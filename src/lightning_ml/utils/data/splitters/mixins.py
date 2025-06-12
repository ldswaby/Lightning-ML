from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from .base_splitter import DataSplitter


class HoldoutMixin(DataSplitter):
    """Mixin implementing the public ``split`` for holdout-style splits."""

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = True,
        random_seed: int = 42,
        **_: Any,
    ) -> None:
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
            raise ValueError(
                f"Sum of train, val, and test ratios must be 1.0. Got {train_ratio + val_ratio + test_ratio:.3f}."
            )
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.shuffle = shuffle
        self.random_seed = random_seed
        super().__init__()

    def split(self, dataset: Dataset, targets: Sequence[Any] | None = None) -> Dict[str, Subset]:
        indices = list(range(len(dataset)))
        train_idx, val_idx, test_idx = self._split_indices(indices, targets)
        return {
            "train": Subset(dataset, train_idx),
            "val": Subset(dataset, val_idx),
            "test": Subset(dataset, test_idx),
        }

    def _split_indices(
        self, indices: Sequence[int], targets: Sequence[Any] | None = None
    ) -> Tuple[Sequence[int], Sequence[int], Sequence[int]]:
        raise NotImplementedError


class RandomHoldoutMixin:
    """Implements random holdout splitting."""

    def _split_indices(
        self, indices: Sequence[int], targets: Sequence[Any] | None = None
    ) -> Tuple[Sequence[int], Sequence[int], Sequence[int]]:
        if targets is not None:
            raise ValueError("targets should not be provided for RandomSplitter")
        n = len(indices)
        train_len = int(self.train_ratio * n)
        val_len = int(self.val_ratio * n)
        test_len = n - train_len - val_len
        if self.shuffle:
            g = torch.Generator().manual_seed(self.random_seed)
            perm = torch.randperm(n, generator=g)
        else:
            perm = torch.arange(n)
        train_idx = perm[:train_len].tolist()
        val_idx = perm[train_len : train_len + val_len].tolist()
        test_idx = perm[train_len + val_len :].tolist()
        return train_idx, val_idx, test_idx


class StratifiedHoldoutMixin:
    """Implements stratified holdout splitting."""

    def _split_indices(
        self, indices: Sequence[int], targets: Sequence[Any] | None = None
    ) -> Tuple[Sequence[int], Sequence[int], Sequence[int]]:
        if targets is None:
            raise ValueError("targets required for stratified split")
        if len(indices) != len(targets):
            raise ValueError("Dataset length and targets length do not match.")
        x_temp, x_test, y_temp, y_test = train_test_split(
            indices,
            targets,
            test_size=self.test_ratio,
            stratify=targets,
            random_state=self.random_seed,
        )
        val_ratio_adj = self.val_ratio / (self.train_ratio + self.val_ratio)
        x_train, x_val, _, _ = train_test_split(
            x_temp,
            y_temp,
            test_size=val_ratio_adj,
            stratify=y_temp,
            random_state=self.random_seed,
        )
        return list(x_train), list(x_val), list(x_test)


class KFoldMixin(DataSplitter):
    """Mixin implementing ``split`` for K-Fold style splits."""

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_seed: int = 42,
        **_: Any,
    ) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_seed = random_seed
        super().__init__()

    def split(self, dataset: Dataset, targets: Sequence[Any] | None = None) -> List[Dict[str, Subset]]:
        indices = list(range(len(dataset)))
        folds = self._split_folds(indices, targets)
        return [
            {"train": Subset(dataset, train), "val": Subset(dataset, val)}
            for train, val in folds
        ]

    def _split_folds(
        self, indices: Sequence[int], targets: Sequence[Any] | None = None
    ) -> List[Tuple[Sequence[int], Sequence[int]]]:
        raise NotImplementedError


class RandomKFoldMixin:
    """Implements standard K-Fold splitting."""

    def _split_folds(
        self, indices: Sequence[int], targets: Sequence[Any] | None = None
    ) -> List[Tuple[Sequence[int], Sequence[int]]]:
        if targets is not None:
            raise ValueError("targets should not be provided for KFoldSplitter")
        kf = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_seed,
        )
        return [
            (train_idx.tolist(), val_idx.tolist())
            for train_idx, val_idx in kf.split(indices)
        ]


class StratifiedKFoldMixin:
    """Implements stratified K-Fold splitting."""

    def _split_folds(
        self, indices: Sequence[int], targets: Sequence[Any] | None = None
    ) -> List[Tuple[Sequence[int], Sequence[int]]]:
        if targets is None:
            raise ValueError("targets required for stratified K-Fold split")
        if len(indices) != len(targets):
            raise ValueError("Dataset length and targets length do not match.")
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_seed,
        )
        return [
            (train_idx.tolist(), val_idx.tolist())
            for train_idx, val_idx in skf.split(indices, targets)
        ]
