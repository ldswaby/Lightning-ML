import inspect

from typing import Any, Dict, Optional, Union

from pytorch_lightning import LightningDataModule
from sklearn.model_selection._split import BaseCrossValidator
from torch.utils.data import DataLoader, Dataset, Subset




class DataModule(LightningDataModule):
    """
    A LightningDataModule that can wrap *any* BaseDataset subclass.
    """

    splits = {"train", "val", "test"}

    def __init__(
        self,
        dataset: Union[Dataset, Dict[str, Dataset], None] = None,
        *,
        splitter: Optional[BaseCrossValidator] = None,
        **dataloader_kwargs,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.splitter = splitter
        self._dataloader_kwargs = dataloader_kwargs
        self.datasets: Dict[str, Dataset] = {}
        if isinstance(dataset, dict):
            self.datasets.update(dataset)

    def define_datasets(self) -> None:
        """Populate ``self.datasets`` if not already provided."""
        if isinstance(self.dataset, dict):
            self.datasets.update(self.dataset)
            return

        if isinstance(self.dataset, Dataset) and self.splitter is not None:
            y = getattr(self.dataset, "targets", None)
            indices = range(len(self.dataset))
            splits = next(self.splitter.split(indices, y))

            if len(splits) == 2:
                train_idx, test_idx = splits
                val_idx = []
            elif len(splits) == 3:
                train_idx, val_idx, test_idx = splits
            else:
                raise ValueError("splitter must yield 2 or 3 index arrays")

            self.datasets["train"] = Subset(self.dataset, train_idx)
            if len(val_idx):
                self.datasets["val"] = Subset(self.dataset, val_idx)
            if len(test_idx):
                self.datasets["test"] = Subset(self.dataset, test_idx)

    # def _logic(self, stage: str):
    #     """TODO: this is a placeholder function, but all data splitting logic should happen within define_Dataset

    #     this logic can be defined
    #     depending e.g. on which script is run (train/test).

    #     Args:
    #         stage (str): _description_
    #     """
    #     if stage == "fit":
    #         # if dataset class has a `train` flag, invoke that = True, alng
    #         # with other args
    #         # e.g. mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
    #         # if doens't have train flag...
    #         pass
    #     if stage == "test":
    #         # train=True
    #         pass
    #     if stage == "predict":
    #         # train=True
    #         pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Create train/val/test ``Dataset`` objects and cache them on ``self.
        datasets``.
        """
        self.define_datasets()
        if not self.datasets:
            raise ValueError("`self.datasets` is empty.")
        if not set(self.datasets.keys()) <= self.splits:
            raise KeyError(f"self.dataset keys must be within {self.splits}.")

    @property
    def dataloader_kwargs(self):
        return self._dataloader_kwargs

    @dataloader_kwargs.setter
    def dataloader_kwargs(self, kwargs: Dict[str, Any]) -> None:
        # Validate that provided kwargs are valid DataLoader parameters
        valid_params = inspect.signature(DataLoader).parameters
        invalid_keys = set(kwargs) - set(valid_params)
        if invalid_keys:
            raise ValueError(f"Invalid DataLoader kwargs: {invalid_keys}")
        self._dataloader_kwargs = kwargs

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["train"], shuffle=True, **self.dataloader_kwargs
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if "val" in self.datasets:
            return DataLoader(self.datasets["val"], **self.dataloader_kwargs)

    def test_dataloader(self) -> Optional[DataLoader]:
        if "test" in self.datasets:
            return DataLoader(self.datasets["test"], **self.dataloader_kwargs)

    @classmethod
    def from_config(cls, cfg: dict) -> "DataModule":
        # TODO
        raise NotImplementedError
