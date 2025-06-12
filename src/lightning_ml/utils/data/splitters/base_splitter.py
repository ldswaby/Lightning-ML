from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset


class DataSplitter(ABC):
    """Abstract base class for dataset splitting strategies."""

    @abstractmethod
    def split(self, dataset: Dataset, *args: Any, **kwargs: Any) -> Any:
        """Split ``dataset`` and return subsets."""
        raise NotImplementedError
