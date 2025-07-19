from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from lightning_ml.core.data import BaseLoader
from lightning_ml.core.utils.loading import (
    IMG_EXTENSIONS,
    has_file_allowed_extension,
    load_image,
)

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


@dataclass
class ImageFiles(BaseLoader):
    """Loads images from filepaths"""

    file_paths: Sequence[str]

    def inputs(self) -> Sequence:
        return [load_image(f) for f in self.file_paths]


@dataclass
class ImageFolder(BaseLoader):
    """Loads all images from specified folder"""

    folder: str

    def inputs(self) -> Sequence:

        file_paths = sorted(  # deterministic
            str(p)
            for p in Path(self.folder).iterdir()
            if has_file_allowed_extension(p.name, IMG_EXTENSIONS)
        )
        return [load_image(f) for f in file_paths]
