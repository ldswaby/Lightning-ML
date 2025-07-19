from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from lightning_ml.core.data.dataset import BaseDataset
from lightning_ml.datasets import LabelledDataset, UnlabelledDataset


@dataclass
class BaseLoader(ABC):
    """Base class for loading data from various sources"""

    @abstractmethod
    def inputs(self) -> Sequence[Any]:
        """Fetch input(s)"""

    def targets(self) -> Optional[Sequence[Any]]:
        """Return target(s), if present"""

    def as_dataset(
        self,
        *,
        dataset_cls: Optional[BaseDataset] = None,
        **dataset_kwargs,
    ):
        """_summary_

        Args:
            dataset_cls (Optional[BaseDataset], optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        inputs = self.inputs()
        targets = self.targets()

        args = [inputs]
        if targets is None:
            dataset_cls = dataset_cls or UnlabelledDataset
        else:
            args.append(targets)
            dataset_cls = dataset_cls or LabelledDataset

        return dataset_cls(*args, **dataset_kwargs)
