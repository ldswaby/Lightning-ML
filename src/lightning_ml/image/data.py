from glob import glob
from typing import Any, Dict
from collections.abc import Callable, Sequence

from lightning_ml.core.data.dataset import BaseDataset
from lightning_ml.datasets.labelled import LabelledDataset

from ..core.data.dtype import DType
from ..core.utils.enums import DataKeys
from ..core.utils.loading import load_image


class Image(DType):

    def process_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Add img metadata

        Args:
            sample (Dict[str, Any]): _description_

        Returns:
            Dict[str, Any]: _description_
        """
        w, h = sample[DataKeys.INPUT].size  # W x H
        if DataKeys.METADATA not in sample:
            sample[DataKeys.METADATA] = {}
        sample[DataKeys.METADATA].update(
            {
                "size": (w, h),
                "height": h,
                "width": w,
            }
        )
        return sample


class ImageFiles(Image):

    def load(self, files: Sequence[str]):
        return [load_image(f) for f in files]


class ImageFolder(ImageFiles):

    def load(self, folder: str):
        return super().load(glob(folder))


# class ImageDataset(LabelledDataset, ImageFiles):
#     """Will have all the functionality of both

#     Args:
#         LabelledDataset (_type_): _description_
#         ImageFiles (_type_): _description_
#     """

#     def __init__(
#         self,
#         folder: str,
#         *,
#         transform: Callable[[Any], Any] | None = None,
#         target_transform: Callable[[Any], Any] | None = None
#     ) -> None:
#         super().__init__(
#             self.load(folder),
#             NotImplemented,
#             transform=transform,
#             target_transform=target_transform,
#         )
