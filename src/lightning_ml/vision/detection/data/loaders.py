# lightning_ml/vision/detection/data/loaders.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Sequence

from lightning_ml.core.utils.loading import (
    IMG_EXTENSIONS,
    has_file_allowed_extension,
    load_image,
)

from ..loaders import ImageFolder  # ← your current class


# ─────────────────────────────────────────────────────────────
# Helper – parse one YOLO-format *.txt label file
# ─────────────────────────────────────────────────────────────
def _parse_yolo_label_file(path: Path) -> list[dict[str, float]]:
    """Return a list of boxes; one dict per object.

    Each line in a YOLO label file:
        <class> <xc> <yc> <w> <h>  (floats in [0, 1] relative coords)
    """
    boxes: list[dict[str, float]] = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls, xc, yc, w, h = map(float, parts)
            boxes.append(
                {
                    "class_id": int(cls),
                    "x_center": xc,
                    "y_center": yc,
                    "width": w,
                    "height": h,
                }
            )
    return boxes


# ─────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────
class YOLOFolder(ImageFolder):
    """YOLO directory-tree loader.

    Expected layout (standard Ultralytics/Roboflow style)::

        root/
          images/            # .jpg/.png but any depth allowed
            train/…
            val/…
          labels/            # parallel text files with the same stem
            train/…
            val/…

    Args
    ----
    root:            Dataset root containing *images/* and *labels/*.
    recursive:       Recurse through sub-dirs inside *images/*.
    images_subdir:   Override if images are not in ``images/``.
    labels_subdir:   Override if labels are not in ``labels/``.
    file_loader:     Function to load an image (default = PIL via ``load_image``).
    img_extensions:  Tuple of allowed extensions.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        *,
        yaml_path="data.yaml",
        images_subdir: str = "images",
        labels_subdir: str = "labels",
        file_loader: Callable[[str | Path], Any] = load_image,
        img_extensions: Sequence[str] = IMG_EXTENSIONS,
    ) -> None:
        # Let ImageFolder set up base fields (root, recursive, etc.)
        super().__init__(root=str(root), recursive=True)

        self.images_dir = Path(self.root) / images_subdir
        self.labels_dir = Path(self.root) / labels_subdir

        # Basic sanity checks
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images folder not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels folder not found: {self.labels_dir}")

    # --------------------------------------------------------
    # Overrides
    # --------------------------------------------------------
    def fetch_samples(self) -> tuple[list[Any], list[list[dict[str, float]]]]:
        """Load every image and its corresponding list of bounding boxes."""
        img_paths = self._iter_image_files()
        images, targets = [], []

        for img_path in img_paths:
            label_path = self._matching_label(img_path)
            images.append(self.file_loader(img_path))
            targets.append(
                _parse_yolo_label_file(label_path) if label_path is not None else []
            )
        return images, targets

    # --------------------------------------------------------
    # Internals
    # --------------------------------------------------------
    def _iter_image_files(self) -> list[Path]:
        """Collect all valid image paths inside *images_dir*."""
        if self.recursive:
            iterator = (
                Path(dirpath) / fname
                for dirpath, _, filenames in os.walk(self.images_dir)
                for fname in sorted(filenames)
            )
        else:
            iterator = sorted(self.images_dir.iterdir())

        return [p for p in iterator if p.is_file() and self._valid_img(str(p))]

    def _matching_label(self, img_path: Path) -> Path | None:
        """Return the matching label file or ``None`` if it doesn’t exist."""
        rel = img_path.relative_to(self.images_dir).with_suffix(".txt")
        cand = self.labels_dir / rel
        return cand if cand.exists() else None
