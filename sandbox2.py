import numpy as np

from src.lightning_ml.core import DataModule
from src.lightning_ml.datasets import (
    ContrastiveLabelledDataset,
    LabelledDataset,
    NumpyLabelledDataset,
    TripletDataset,
)

# Set random seed for reproducibility
np.random.seed(42)

# Create 100 samples with 2 features each
X = np.random.randn(100, 2)

# Create labels based on a linear decision boundary (just for demo)
# Label = 1 if x1 + x2 > 0, else 0
y = (X[:, 0] + X[:, 1] > 0).astype(int)


dm = DataModule(
    dataset_cls=TripletDataset,
    dataset_kwargs={"inputs": X, "targets": y},
    dataloader_kwargs={"pin_memory": True},
)

dl = dm.train_dataloader()
# ds = TripletDataset(X, y)
breakpoint()
print(dataset.sample_keys)
