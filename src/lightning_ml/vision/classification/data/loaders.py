"""Module for image-based data loading.

Defines data loader classes for loading images from file paths or folders,
with simple interfaces to return loaded image objects.
"""

from collections.abc import Callable
from pathlib import Path

from ...loaders import ImageFolder


class ImageClassificationFolder(ImageFolder):
    """Loads all images from specified folder"""

    def __init__(self, root: str):
        """Initialize the ImageFolder loader.

        Args:
            root: Directory containing image files.
        """
        super().__init__(
            root=root,
            recursive=True,
            load_targets=self.subdirs_as_classes,
        )

    def subdirs_as_classes(self, directory: str | Path) -> list[str]:
        """
        Default loader for targets: for each valid file under root,
        returns the upper directory name (the first path component under root).
        """
        root = Path(directory)
        files = self._iter_valid_files(root)
        return [p.relative_to(root).parts[0] for p in files]
