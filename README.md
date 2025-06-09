# Lightning-ML

Lightning-ML is a lightweight research framework built on top of [PyTorch Lightning](https://www.pytorchlightning.ai/). It provides a small collection of abstractions that help organise datasets, models and training logic while remaining compatible with the Lightning ecosystem.

## Features

- **Modular learners** – `Learner` classes wrap your `nn.Module` and define the training, validation and prediction steps.
- **Prediction utilities** – `Predictor` objects convert raw model outputs into task-specific predictions (e.g. class indices).
- **Dataset building blocks** – reusable dataset mix-ins and ready-to-use in-memory datasets for labelled, unlabelled and contrastive setups.
- **Generic datamodule** – the `DataModule` can turn any dataset into Lightning data loaders with optional train/val/test splits.
- **Model registry** – simple registry mechanism for declaring and retrieving models by name.
- **Project orchestration** – the `Project` class wires together a learner and a Lightning `Trainer` instance for a concise training loop.

## Repository layout

```
src/lightning_ml/
├── core/        # base classes: Learner, Predictor, Dataset, DataModule, Project
├── datasets/    # dataset mix-ins and concrete dataset implementations
├── learners/    # common learning paradigms (Supervised, Unsupervised, Contrastive)
├── models/      # model registry and example model
├── predictors/  # output post-processing utilities
└── utils/       # helper functions (registries, class binding, ...)
```

An example classification setup is provided in `sandbox.py`.

## Quick example

Below is a minimal illustration of how the pieces fit together:

```python
import torch
from torch import nn
from torch.optim import Adam
from pytorch_lightning import Trainer

from lightning_ml.core import DataModule, Project
from lightning_ml.datasets import NumpyLabelledDataset
from lightning_ml.learners import Supervised
from lightning_ml.models import MyCustomModel
from lightning_ml.predictors import Classification

# Dataset and datamodule
dm = DataModule(
    dataset_cls=NumpyLabelledDataset,
    dataset_kwargs={"inputs": "x.npy", "targets": "y.npy"},
    dataloader_kwargs={"batch_size": 32}
)

# Model and learner
model = MyCustomModel(input_size=10, hidden_size=32, output_size=3)
learner = Supervised(
    model=model,
    optimizer=Adam(model.parameters()),
    data=dm,
    criterion=nn.CrossEntropyLoss(),
)
learner.predictor = Classification()

# Train and evaluate
project = Project(learner, trainer=Trainer())
project.train()
```

The framework stays intentionally small and flexible so you can plug in any PyTorch modules, metrics or Lightning `Trainer` configuration.

## License

This project is released under the terms of the MIT license. See [LICENSE](LICENSE) for details.
