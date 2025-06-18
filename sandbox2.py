import numpy as np

from src.lightning_ml.core import DataModule
from src.lightning_ml.datasets import (
    ContrastiveLabelledDataset,
    LabelledDataset,
    NumpyLabelledDataset,
    TripletDataset,
)


class NumpyDataModule(DataModule):

    def define_datasets(self) -> None:
        self.datasets["train"] = NumpyLabelledDataset("_data/X.npy", "_data/y.npy")
        breakpoint()
        return


data = NumpyDataModule()
data.setup()

breakpoint()
