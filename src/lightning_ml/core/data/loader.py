from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from lightning_ml.core.data.dataset import BaseDataset
from lightning_ml.data.datasets import LabelledDataset, UnlabelledDataset


@dataclass
class BaseLoader(ABC):
    """Base class for loading data from various sources"""

    @abstractmethod
    def load_inputs(self) -> Sequence[Any]:
        """Fetch input(s)"""

    def load_targets(self) -> Sequence[Any] | None:
        """Return target(s), if present"""
        return None

    def as_dataset(
        self,
        *,
        dataset_cls: BaseDataset | None = None,
        **dataset_kwargs,
    ):
        """_summary_

        Args:
            dataset_cls (Optional[BaseDataset], optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        inputs = self.load_inputs()
        targets = self.load_targets()

        args = [inputs]
        if targets is None:
            dataset_cls = dataset_cls or UnlabelledDataset
        else:
            args.append(targets)
            dataset_cls = dataset_cls or LabelledDataset

        return dataset_cls(*args, **dataset_kwargs)
