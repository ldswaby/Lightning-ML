"""Common dataset implementations that load from disk, leveraging existing mix-ins."""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from .labelled import LabelledDataset
from .unlabelled import UnlabelledDataset

__all__ = [
    "NumpyUnlabelledDataset",
    "NumpyLabelledDataset",
    "CSVDataset",
    "ImageFolderDataset",
]


class NumpyUnlabelledDataset(UnlabelledDataset):
    """Unlabelled dataset backed by a NumPy array or ``.npy`` file."""

    def __init__(
        self,
        inputs: str | Sequence[Any] | np.ndarray,
        *,
        transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        if isinstance(inputs, str):
            inputs = np.load(inputs)
        super().__init__(inputs, transform=transform)


class NumpyLabelledDataset(LabelledDataset):
    """Labelled dataset backed by NumPy arrays or ``.npy`` files."""

    def __init__(
        self,
        inputs: str | Sequence[Any] | np.ndarray,
        targets: str | Sequence[Any] | np.ndarray,
        *,
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        if isinstance(inputs, str):
            inputs = np.load(inputs)
        if isinstance(targets, str):
            targets = np.load(targets)
        super().__init__(
            inputs,
            targets,
            transform=transform,
            target_transform=target_transform,
        )


class CSVDataset(LabelledDataset):
    """Labelled dataset loaded from a CSV file."""

    def __init__(
        self,
        csv_path: str,
        input_cols: Sequence[str],
        target_col: str,
        *,
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
        pandas_kwargs: Optional[dict] = None,
    ) -> None:
        df = pd.read_csv(csv_path, **(pandas_kwargs or {}))
        inputs = df[input_cols].values
        targets = df[target_col].values
        super().__init__(
            inputs, targets, transform=transform, target_transform=target_transform
        )


class ImageFolderDataset(LabelledDataset):
    """Simple image classification dataset from a folder structure."""

    def __init__(
        self,
        root: str,
        *,
        transform: Optional[Callable[[Image.Image], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
        image_extensions: Sequence[str] | None = None,
    ) -> None:
        self.root = root
        self.image_transform = transform or (lambda x: x)
        image_extensions = image_extensions or [".png", ".jpg", ".jpeg", ".bmp", ".gif"]

        classes = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        inputs: list[str] = []
        targets: list[int] = []
        for cls_name in classes:
            class_dir = os.path.join(root, cls_name)
            for fname in os.listdir(class_dir):
                if any(fname.lower().endswith(ext) for ext in image_extensions):
                    inputs.append(os.path.join(class_dir, fname))
                    targets.append(self.class_to_idx[cls_name])

        super().__init__(
            inputs, targets, transform=None, target_transform=target_transform
        )

    def get_input(self, idx: int) -> Any:
        path = self._inputs[idx]
        img = Image.open(path).convert("RGB")
        return self.image_transform(img)
