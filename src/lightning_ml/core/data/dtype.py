from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

from lightning_ml.core.data.dataset import BaseDataset, LabelledDatasetBase, TargetMixin


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


class YOLOFolder(LabelledDatasetBase):
    """Base class for loading data from disk into Sequence[Any]

    Args:
        ABC (_type_): _description_
    """

    def __init__(self, root: str) -> None:
        # glob stuff here (train val test etc)
        # lists of file paths
        super().__init__()

    def get_sample(self, idx):
        # load file and return raw here
        pass

    def get_target(self, idx) -> Dict[str, Any]:
        """Process sample (e.g. add metadata)

        Returns:
            Sequence[Any]: _description_
        """
        return ObjDetTarget[idx]  # bboxes, labels


class Target(TargetMixin):
    """Base class for target dicts (e.g. {bboxes.., labels...})"""

    sample_keys: List[str] = []

    def get_target(self, idx):
        return {k: getattr(self, f"get_{k}")(idx) for k in self.sample_keys}


class ObjDetTarget(Target):
    """Base class for loading data from disk into Sequence[Any]

    Args:
        ABC (_type_): _description_
    """

    sample_keys = [
        "bboxes",
        "labels",
    ]  # this needs these fns, but YOLOfoldershouldn't, as it automatically inherits from here. however, bc sample keys joins for all, it will return one long dict, but tgt dict and stuff above should be seperate

    def get_bboxes(self, idx):
        pass

    def get_labels(self, idx):
        pass
