from __future__ import annotations

from typing import Any, Optional, Type

try:
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
except Exception as e:  # pragma: no cover - import dependency
    raise ImportError("PyTorch Lightning and torch are required") from e

from .dataset import BaseDataset


class DataModule(pl.LightningDataModule):
    """Minimal Lightning ``DataModule`` for a single dataset."""

    def __init__(
        self,
        dataset_cls: Type[BaseDataset],
        *,
        dataset_kwargs: Optional[dict[str, Any]] = None,
        dataloader_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.dataset_cls = dataset_cls
        self.dataset_kwargs = dataset_kwargs or {}
        self.dataloader_kwargs = dataloader_kwargs or {}
        self.dataset: Optional[BaseDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = self.dataset_cls(**self.dataset_kwargs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, **self.dataloader_kwargs)
