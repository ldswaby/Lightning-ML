from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from lightning_ml.core.data.dataset import BaseDataset

from .datasets import LabelledDataset, UnlabelledDataset


@dataclass
class BaseLoader(ABC):
    """Base class for loading data from various sources"""

    @abstractmethod
    def load_inputs(self) -> Sequence[Any]:
        """Fetch input(s)"""

    def load_targets(self) -> Optional[Sequence[Any]]:
        """Return target(s), if present"""

    # @abstractmethod
    # def fetch_samples(self) -> tuple[Sequence[Any], Optional[Sequence[Any]]]:
    #     """Fetch input(s) and (optionally) targets"""

    def as_dataset(
        self,
        *,
        dataset_cls: Optional[BaseDataset] = None,
        **dataset_kwargs,
    ):
        """Return an instantiated dataset containing the loaded samples.

        Args:
            dataset_cls: Optional dataset type to use. If ``None`` then
                :class:`UnlabelledDataset` is used when ``fetch_samples`` does
                not return targets, otherwise :class:`LabelledDataset`.

        Returns:
            The created dataset instance.
        """
        inputs, targets = self.fetch_samples()

        args = [inputs]
        if targets is None:
            dataset_cls = dataset_cls or UnlabelledDataset
        else:
            args.append(targets)
            dataset_cls = dataset_cls or LabelledDataset

        return dataset_cls(*args, **dataset_kwargs)
