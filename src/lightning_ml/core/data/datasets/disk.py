"""Common dataset implementations that load from disk, leveraging existing mix-ins."""

from __future__ import annotations

import os
from typing import Any, Optional
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from ...utils.enums import Registries
from ...utils.registry import register
from .labelled import LabelledDataset
from .unlabelled import UnlabelledDataset

__all__ = [
    "NumpyUnlabelledDataset",
    "NumpyLabelledDataset",
    "CSVDataset",
    "ImageFolderDataset",
]


@register(Registries.DATASET)
class NumpyUnlabelledDataset(UnlabelledDataset):
    """Unlabelled dataset backed by a NumPy array or ``.npy`` file."""

    def __init__(
        self,
        inputs: str | Sequence[Any] | np.ndarray,
        *,
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        if isinstance(inputs, str):
            inputs = np.load(inputs)
        super().__init__(inputs, transform=transform)


@register(Registries.DATASET)
class NumpyLabelledDataset(LabelledDataset):
    """Labelled dataset backed by NumPy arrays or ``.npy`` files."""

    def __init__(
        self,
        inputs: str | Sequence[Any] | np.ndarray,
        targets: str | Sequence[Any] | np.ndarray,
        *,
        transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[Any], Any] | None = None,
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


@register(Registries.DATASET)
class CSVDataset(LabelledDataset):
    """Labelled dataset loaded from a CSV file."""

    def __init__(
        self,
        csv_path: str,
        input_cols: Sequence[str],
        target_col: str,
        *,
        transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[Any], Any] | None = None,
        pandas_kwargs: dict | None = None,
    ) -> None:
        df = pd.read_csv(csv_path, **(pandas_kwargs or {}))
        inputs = df[input_cols].values
        targets = df[target_col].values
        super().__init__(
            inputs, targets, transform=transform, target_transform=target_transform
        )


@register(Registries.DATASET)
class ImageFolderDataset(LabelledDataset):
    """Simple image classification dataset from a folder structure."""

    def __init__(
        self,
        root: str,
        *,
        transform: Callable[[Image.Image], Any] | None = None,
        target_transform: Callable[[Any], Any] | None = None,
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
