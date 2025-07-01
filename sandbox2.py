import numpy as np
from pytorch_lightning import Trainer
from sklearn.model_selection import KFold, ShuffleSplit
from torch import Tensor, nn, optim
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall

from src.lightning_ml.core import DataModule, School
from src.lightning_ml.datasets import (
    ContrastiveLabelledDataset,
    LabelledDataset,
    NumpyLabelledDataset,
    TripletDataset,
)
from src.lightning_ml.learners import Supervised
from src.lightning_ml.models import MyCustomModel
from src.lightning_ml.utils.data import validation_split

trainset = NumpyLabelledDataset("_data/X.npy", "_data/y.npy")
testset = NumpyLabelledDataset("_data/X.npy", "_data/y.npy")

shuffle_split = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

splits = validation_split(trainset, shuffle_split)


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

for fold_idx, (trainset, valset) in enumerate(splits):
    model = MyCustomModel(input_size=4, hidden_size=64, output_size=NUM_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    learner = Supervised(
        model=model,
        optimizer=optimizer,
        # data_module=data,x
        criterion=criterion,
        metrics=metrics,
        # scheduler: Optional[_LRScheduler] = None,
        # predictor: Optional[Predictor] = None,
    )

    school = School(learner=learner, trainer=Trainer())
    school.train()
