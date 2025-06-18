import numpy as np
from torch import Tensor, nn, optim
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall

from src.lightning_ml.core import DataModule
from src.lightning_ml.datasets import (
    ContrastiveLabelledDataset,
    LabelledDataset,
    NumpyLabelledDataset,
    TripletDataset,
)
from src.lightning_ml.learners import Supervised
from src.lightning_ml.models import MyCustomModel


class NumpyDataModule(DataModule):

    def define_datasets(self) -> None:
        self.datasets["train"] = NumpyLabelledDataset("_data/X.npy", "_data/y.npy")
        return


data = NumpyDataModule()


NUM_CLASSES = 2
# Define a set of metrics to track
classification_metrics = MetricCollection(
    {
        "accuracy": Accuracy(task="multiclass", num_classes=NUM_CLASSES),
        "precision": Precision(
            task="multiclass", num_classes=NUM_CLASSES, average="macro"
        ),
        "recall": Recall(task="multiclass", num_classes=NUM_CLASSES, average="macro"),
        "f1": F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro"),
    }
)

# Create a dict mapping each stage to its MetricCollection
metrics = {
    "train": classification_metrics.clone(prefix="train_"),
    "val": classification_metrics.clone(prefix="val_"),
    "test": classification_metrics.clone(prefix="test_"),
}

model = MyCustomModel(input_size=2, hidden_size=64, output_size=NUM_CLASSES)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

learner = Supervised(
    model=model,
    optimizer=optimizer,
    data_module=data,
    criterion=criterion,
    metrics=metrics,
    # scheduler: Optional[_LRScheduler] = None,
    # predictor: Optional[Predictor] = None,
)

breakpoint()
