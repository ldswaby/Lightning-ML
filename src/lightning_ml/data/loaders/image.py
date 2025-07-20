"""Module for image-based data loading.

Defines data loader classes for loading images from file paths or folders,
with simple interfaces to return loaded image objects.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Sequence

from lightning_ml.core.data import BaseLoader
from lightning_ml.core.utils.loading import (
    IMG_EXTENSIONS,
    has_file_allowed_extension,
    load_image,
)
from lightning_ml.data.loaders.folder import Folder, subdirs_as_classes

# class Image(DType):

#     def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
#         """Add img metadata

#         Args:
#             sample (Dict[str, Any]): _description_

#         Returns:
#             Dict[str, Any]: _description_
#         """
#         w, h = sample[DataKeys.INPUT].size  # W x H
#         if DataKeys.METADATA not in sample:
#             sample[DataKeys.METADATA] = {}
#         sample[DataKeys.METADATA].update(
#             {
#                 "size": (w, h),
#                 "height": h,
#                 "width": w,
#             }
#         )
#         return sample


class ImageFiles(BaseLoader):
    """Loads images from filepaths"""

    def __init__(self, file_paths: Sequence[str]):
        """Initialize the ImageFiles loader.

        Args:
            file_paths (Sequence[str]): List of file paths to load images from.
        """
        self.file_paths = file_paths

    def fetch_samples(self) -> tuple[Sequence[Any], Sequence[Any] | None]:
        return [load_image(f) for f in self.file_paths], None


class ImageFolder(Folder):
    """Loads all images from specified folder"""

    def __init__(
        self,
        root: str,
        recursive: bool = False,
        find_classes: Callable[
            [str | Path], tuple[list[str], dict[str, int]]
        ] = subdirs_as_classes,
    ):
        """Initialize the ImageFolder loader.

        Args:
            folder (str): Path to the directory containing image files.
        """
        super().__init__(
            root=root,
            file_loader=load_image,
            extensions=IMG_EXTENSIONS,
            recursive=recursive,
            find_classes=find_classes,
        )
