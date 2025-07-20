import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Dict, Optional

from lightning_ml.core.data import BaseLoader
from lightning_ml.core.utils.loading import has_file_allowed_extension


class Folder(BaseLoader):
    """
    Folder data loader.

    Loads data from a directory structure where each subdirectory is a class.

    Args:
        root (str): Root directory path containing class subdirectories.
        file_loader (Callable[[str | Path], Any]): Function to load a file given its path.
        extensions (Sequence[str]): Allowed file extensions.
        is_valid_file (Callable[[str], bool] | None): Optional function to validate filenames.
        load_targets (Callable[[str | Path], tuple[list[str], dict[str, int]]]): Function to find class folders and map them to indices.
    """

    def __init__(
        self,
        root: str,
        file_loader: Callable[[str | Path], Any],
        extensions: tuple[str, ...] | None = None,
        is_valid_file: Callable[[str], bool] | None = None,
        load_targets: Callable[[str | Path], Optional[Sequence[Any]]] | None = None,
        recursive: bool = False,
    ):
        """Initialize the Folder data loader.

        Args:
            root (str): Path to the root directory containing class subdirectories.
            file_loader (Callable[[str | Path], Any]): Function to load a file given its path.
            extensions (tuple[str, ...] | None): Optional tuple of allowed file extensions.
            is_valid_file (Callable[[str], bool] | None): Optional function to validate filenames.
            load_targets (Callable[[str | Path], tuple[list[str], dict[str, int]]]): Function to find all targets from root
            recursive (bool): If True, traverse directories recursively.

        Raises:
            ValueError: If both 'extensions' and 'is_valid_file' are specified.
        """
        self.root = root
        self.file_loader = file_loader
        self.extensions = extensions
        self.is_valid_file = is_valid_file
        self._load_targets = load_targets or (lambda _: None)
        self.recursive = recursive

        # Validate filter configuration
        if self.extensions is not None and self.is_valid_file is not None:
            raise ValueError("Cannot specify both 'extensions' and 'is_valid_file'")

        # If extensions provided but no custom validator, use extension-based filter
        if self.extensions is not None:
            exts = self.extensions
            self.is_valid_file = lambda x: has_file_allowed_extension(x, exts)

        # Default behaviour: assume all files valid
        self.is_valid_file = self.is_valid_file or (lambda _: True)

    def _iter_valid_files(self, directory: Path) -> list[Path]:
        """Return an unsorted list of valid file paths found inside *directory*.

        The search respects the ``self.recursive`` flag and uses
        ``self.is_valid_file`` if defined; otherwise, it accepts every file.
        """

        if self.recursive:
            iterator = (
                Path(dirpath) / fname
                for dirpath, _, filenames in os.walk(directory)
                for fname in filenames
            )
        else:
            # Return only valid files in root
            iterator = (entry for entry in directory.iterdir() if entry.is_file())

        return [p for p in iterator if self.is_valid_file(str(p))]

    def load_inputs(self) -> Sequence[Any]:
        paths = self._iter_valid_files(Path(self.root))
        return [self.file_loader(p) for p in paths]

    def load_targets(self) -> Sequence[Any] | None:
        """NOTE defaults to return None

        Returns:
            Sequence[Any] | None: _description_
        """
        return self._load_targets(self.root)

    # def fetch_samples(self) -> tuple[Sequence[Any], Sequence[Any] | None]:
    #     """Fetch input samples and their targets (if any)."""
    #     root_path = Path(self.root)
    #     classes = self.load_targets(root_path)

    #     # Case 1 – unlabelled dataset (no class sub‑directories)
    #     if not classes:
    #         paths = self._iter_valid_files(root_path)
    #         return [self.file_loader(p) for p in paths], None

    #     # Case 2 – labelled dataset
    #     inputs: list[Any] = []
    #     targets: list[str] = []  # TODO: change for target formatter

    #     for class_name in classes:
    #         class_dir = root_path / class_name
    #         if not class_dir.is_dir():
    #             continue  # Skip if the expected folder is missing

    #         for path in self._iter_valid_files(class_dir):
    #             inputs.append(self.file_loader(path))
    #             targets.append(class_name)

    #     return inputs, targets
