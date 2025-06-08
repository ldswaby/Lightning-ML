from typing import Dict

import pytorch_lightning as pl

# #############  dummy supervised‑classification setup  #############
import torch
from pytorch_lightning import Trainer
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, MetricCollection

from src.lightning_ml.core.project import Project
from src.lightning_ml.core import PredictorWrapper
from src.lightning_ml.learners import Supervised
from src.lightning_ml.predictors import Classification

NUM_CLASSES = 3
INPUT_SHAPE = (1, 28, 28)  # (C, H, W)


class DummyClassificationDataset(Dataset):
    """Generates random images and labels; returns dicts with 'input', 'target'."""

    def __init__(self, length: int = 120):
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        x = torch.randn(INPUT_SHAPE)  # random image
        y = torch.randint(0, NUM_CLASSES, ())  # scalar label
        return {"input": x, "target": y}


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 16):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: str | None = None):
        self.train_set = DummyClassificationDataset(120)
        self.val_set = DummyClassificationDataset(40)
        self.test_set = DummyClassificationDataset(40)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)


# simple classifier: flatten → linear
class FlatClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        in_features = int(torch.prod(torch.tensor(INPUT_SHAPE)))
        self.net = nn.Linear(in_features, NUM_CLASSES)

    def forward(self, x):
        """Flatten (B, C, H, W) input and run the linear head.

        Accepts either:
        * the full batch dict produced by the dataset (`{"input": ..., "target": ...}`)
        * or the raw image tensor directly.
        """
        if isinstance(x, dict):  # dataset passes the whole dict
            x = x["input"]
        x = x.view(x.size(0), -1)
        return self.net(x)


model: nn.Module = FlatClassifier()
data: pl.LightningDataModule = DummyDataModule()

criterion: nn.Module = nn.CrossEntropyLoss()
metrics: Dict[str, MetricCollection] = {
    s: MetricCollection({"acc": Accuracy(task="multiclass", num_classes=NUM_CLASSES)})
    for s in ["train", "val", "test"]
}

optimizer: Optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = None
# #############  end dummy setup #############


learner = Supervised(
    model=model,
    optimizer=optimizer,
    data=data,
    criterion=criterion,
    metrics=metrics,
    scheduler=None,  # TODO
)

predictor = Classification()
student = PredictorWrapper(learner, predictor)


project = Project(student=student, trainer=Trainer())

project.train()

# QUESTION: I don't like either of the above.
# Why can't I define:
# 1. Learner (with optional ``PredictorWrapper`` to combine a predictor later)
# 3. Trainer (pytorch_lighting.Trainer)
# 4. Project object which combines all the above
