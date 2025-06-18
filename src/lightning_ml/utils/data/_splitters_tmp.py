"""Data split utilities that extend scikit-learn splitters with a **validation** subset.

This module defines :class:`TrainTestValSplitter`, a lightweight wrapper around any
scikit-learn :class:`sklearn.model_selection.BaseCrossValidator` that leaves the
training indices untouched but subdivides the held-out indices into *validation*
and *test* parts.
"""

from typing import Iterator, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection._split import *
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.utils import check_random_state


class TrainTestValSplitter(BaseCrossValidator):
    """Wrap any scikit-learn splitter and add a validation split.

    The wrapper keeps the training indices produced by the underlying splitter
    unchanged but further divides its held-out indices into **validation** and
    **test** subsets.

    Args:
        base_splitter (BaseCrossValidator): A scikit-learn splitter that yields
            ``(train_idx, test_idx)`` pairs.
        val_size (float | int, optional): Size of the validation subset relative
            to the original ``test_idx``.
            * If ``float`` (``0 < val_size < 1``) it is treated as a proportion
              of ``test_idx``.
            * If ``int`` (``>= 1``) it is the absolute number of samples.
            Defaults to ``0.5``.
        shuffle (bool, optional): Whether to shuffle the held-out indices before
            partitioning them into validation and test. Defaults to ``True``.
        random_state (int | np.random.RandomState | None, optional): Seed or RNG
            for reproducible shuffling. Defaults to ``None``.

    Raises:
        TypeError: If inputs are of invalid type.
        ValueError: If ``val_size`` is outside the allowed range or would create
            empty validation or test subsets.

        Note:
            If the underlying splitter is *stratified* (e.g. ``StratifiedKFold``,
            ``StratifiedShuffleSplit``) **and** you pass the target labels ``y``
            to :meth:`split`, the wrapper preserves that stratification in the
            secondary validation/test split as well.

    Examples:
        >>> from sklearn.model_selection import ShuffleSplit
        >>> import numpy as np
        >>> splitter = TrainTestValSplitter(
        ...     ShuffleSplit(n_splits=2, test_size=0.2, random_state=0),
        ...     val_size=0.5,
        ...     random_state=0,
        ... )
        >>> X = np.arange(100)
        >>> for tr, val, te in splitter.split(X):
        ...     print(len(tr), len(val), len(te))
        80 10 10
        80 10 10
    """

    def __init__(
        self,
        base_splitter: BaseCrossValidator,
        *,
        val_size: Union[float, int] = 0.5,
        shuffle: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        if not isinstance(base_splitter, BaseCrossValidator):
            raise TypeError("base_splitter must inherit from BaseCrossValidator")

        if isinstance(val_size, float):
            if not 0 < val_size < 1:
                raise ValueError("val_size as float must be in (0, 1)")
        elif isinstance(val_size, int):
            if val_size < 1:
                raise ValueError("val_size as int must be >= 1")
        else:
            raise TypeError("val_size must be float or int")

        self.base_splitter = base_splitter
        self.val_size = val_size
        self.shuffle = shuffle
        self.random_state = check_random_state(random_state)

        # Detect whether the underlying splitter uses stratification
        self._stratified = self._is_stratified_splitter(base_splitter)

    # ------------------------------------------------------------------#
    # Helpers                                                            #
    # ------------------------------------------------------------------#
    @staticmethod
    def _is_stratified_splitter(splitter: BaseCrossValidator) -> bool:
        """Heuristic to detect whether *splitter* enforces class stratification."""
        return splitter.__class__.__name__.lower().startswith("stratified")

    # ---------------------------------------------------------------------
    # Standard *sklearn* API
    # ---------------------------------------------------------------------
    def get_n_splits(
        self, X: Sequence | None = None, y=None, groups=None
    ) -> int:  # noqa: D401,E501
        """Return the number of splitting iterations.

        Args:
            X (Sequence | None, optional): Feature matrix or indices.
            y (Sequence | None, optional): Target labels. Defaults to ``None``.
            groups (Sequence | None, optional): Group labels. Defaults to ``None``.

        Returns:
            int: Number of splits produced by the underlying splitter.
        """
        return self.base_splitter.get_n_splits(X, y, groups)

    def split(
        self, X: Sequence, y=None, groups=None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, val_idx, test_idx)`` index tuples.

        Args:
            X (Sequence): Feature matrix or indices.
            y (Sequence | None, optional): Target labels. Defaults to ``None``.
            groups (Sequence | None, optional): Group labels. Defaults to ``None``.

        Yields:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Train, validation and
            test indices for each split.
        """
        for train_idx, holdout_idx in self.base_splitter.split(X, y, groups):
            holdout_idx = np.asarray(holdout_idx, dtype=int)

            # Determine validation size
            if isinstance(self.val_size, float):
                n_val = int(np.floor(len(holdout_idx) * self.val_size))
            else:  # int
                n_val = int(self.val_size)

            if n_val == 0 or n_val >= len(holdout_idx):
                raise ValueError(
                    f"val_size={self.val_size!r} results in empty validation or test "
                    f"set for split of length {len(holdout_idx)}."
                )

            # ------------------------------------------------------------#
            # Option A: keep stratification if the original splitter was  #
            # stratified and y is supplied.                               #
            # ------------------------------------------------------------#
            if self._stratified and y is not None:
                stratify_y = np.asarray(y)[holdout_idx]
            else:
                stratify_y = None

            if stratify_y is not None:
                # train_test_split returns (train_subset, test_subset)
                val_idx, test_idx = train_test_split(
                    holdout_idx,
                    test_size=len(holdout_idx) - n_val,
                    random_state=self.random_state,
                    shuffle=self.shuffle,
                    stratify=stratify_y,
                )
            else:
                # Fall back to pure random shuffling
                if self.shuffle:
                    self.random_state.shuffle(holdout_idx)
                val_idx = holdout_idx[:n_val]
                test_idx = holdout_idx[n_val:]

            yield np.asarray(train_idx, dtype=int), val_idx, test_idx

    # ------------------------------------------------------------------
    # Misc utilities
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}(base_splitter={self.base_splitter}, "
            f"val_size={self.val_size}, shuffle={self.shuffle})"
        )

    def __getattr__(self, item):  # pragma: no cover
        """Delegate missing attributes to *base_splitter* (e.g. n_splits)."""
        return getattr(self.base_splitter, item)


# -----------------------------------------------------------------------------#
# Public API
# -----------------------------------------------------------------------------#
__all__: list[str] = ["TrainTestValSplitter", "ValidationSplitter"]

# Backwards‑compatibility alias
ValidationSplitter = TrainTestValSplitter
