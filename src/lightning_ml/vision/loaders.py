"""Module for image-based data loading.

Defines data loader classes for loading images from file paths or folders,
with simple interfaces to return loaded image objects.
"""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, Sequence

from lightning_ml.core.data import BaseLoader
from lightning_ml.core.data.loaders.folder import Folder
from lightning_ml.core.utils.loading import IMG_EXTENSIONS, load_image

# def subdirs_as_classes(directory: str | Path) -> list[str]:
#     """Finds the class folders in a dataset. Assumes first=layer subdirs."""
#     return sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())


class ImageFiles(BaseLoader):
    """Loads images from filepaths"""

    def __init__(self, file_paths: Sequence[str]):
        """Initialize the ImageFiles loader.

        Args:
            file_paths (Sequence[str]): List of file paths to load images from.
        """
        self.file_paths = file_paths

    def load_inputs(self) -> Sequence[Any]:
        return [load_image(f) for f in self.file_paths]


class ImageFolder(Folder):
    """Loads all images from specified folder"""

    def __init__(
        self,
        root: str,
        recursive: bool = False,
        load_targets: Callable[[str | Path], Optional[Sequence[Any]]] | None = None,
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
            load_targets=load_targets,
        )
