from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence


class DType(ABC):
    """Base class for loading data from disk into Sequence[Any]

    Args:
        ABC (_type_): _description_
    """

    @abstractmethod
    def load(self, *args, **kwargs) -> Sequence[Any]:
        """Load samples from disk

        Returns:
            Sequence[Any]: _description_
        """

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process sample (e.g. add metadata)

        Returns:
            Sequence[Any]: _description_
        """
        return sample
