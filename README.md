# Lightning-ML

## TODO:
* A given DataModule will
  * mix (somehow) with
    *  `Input` for custom loading from disk functionality (i.e. into inputs and, optionally, targets)
    *  `Dataset` for the sample building behaviour (e.g. Labelled, Contrastive) -> inputs, targets, metadata etc
* Have various factory methods
* `Input` funcitons load sequences (for dataset _inputs) from various formats


* Take the following from `lihgtning-flash`:
  * `providers`
  * `DataModule`
  *



```





*NOTE*: This is a personal repository designed to serve as a quick-start framework for future machine learning (ML) projects.

Built on top of [PyTorch Lightning](https://www.pytorchlightning.ai/), it provides modular libraries for all major components of the standard ML pipeline: datasets, models, loss functions, optimizers, and evaluation metrics. These are automatically piped together in a lightweight, easily configurable manner (either programmatically or through config files), enabling rapid development and deployment of ML projects from scratch.

The codebase structure is based on a simple conceptualisation of end-to-end ML projects consisting of:
1. A `Learner`: a *learning problem*, consisting of learning material (`data`), a learning agent (`model`), and a learning method (`loss` + `optimizer`).
2. A `Predictor`: A *downstream task* to which the learning agent applies its newly aquired knowledge (e.g. `Classification`, `Multiregression`). This more lightweight component will typically just add some additional post-processing steps to `Learner`'s outputs (e.g. softmax + argmax for classification).

We divorce the two to preserve the ability to use models trained in one way to be ultimately used in others (e.g. a self-supervised contrastive learner used for classification)

The true power of this repository is that it chains togetehr libraries from other popular ML packages. E.g., on top of any custom defined, you have automatic access to:

* `torchvision` and `torchaudio` datasets and models
* `torch.nn` loss functions
* `sklearn` data splitters
* `torchmetrics` evaluation metrics


To define a learner, you have to define the code that pipes a batch to model output(s) (`batch_forward`), and how to get from these to a loss value (`compute_loss`)

## Features

- **Modular learners** – `Learner` classes wrap your `nn.Module` and define the training, validation and prediction steps.
- **Prediction utilities** – `Predictor` objects convert raw model outputs into task-specific predictions (e.g. class indices).
- **Dataset building blocks** – reusable dataset mix-ins and ready-to-use in-memory datasets for labelled, unlabelled and contrastive setups.
- **Generic datamodule** – the `DataModule` can turn any dataset into Lightning data loaders with optional train/val/test splits.
- **Component registries** – simple registry mechanism for declaring and retrieving
  datasets, learners, models and predictors by name.
- **Torchvision integration** – utility functions to auto-register torchvision models and datasets.
- **Project orchestration** – the `Project` class wires together a learner and a Lightning `Trainer` instance for a concise training loop.

## Data
The two most common workflows with respect to splitting up data for machine learning are simple hold-out and cross-validation.

I want my codebase to support both.

### TODO:
* Copy the dataset structure from `lightning-flash`


1. Starting point: load the dataset into a format compatible with my API. This may yield a single full `train` dataset or a `train` and a `test`.
   1. If a single dataset, then `test` must be split off manually here.
2. Pass the `train` dataset to a `utils.data.validation_split`  which will return a list of dicts: `[{"train": ..., "val": ...}, ...]` for the train+validation loop (that we will later iterate through). This enables standard holdout (e.g. `ShuffleSplit`) or CV (e.g. `KFold`)
3. Run train + validation loop by iterating over the list from step 2 and aggregating results (see `sandbox2.py`)






## Repository layout

```
src/lightning_ml/
├── core/        # base classes: Learner, Predictor, Dataset, DataModule, Project
├── datasets/    # dataset mix-ins and concrete dataset implementations
├── learners/    # common learning paradigms (Supervised, Unsupervised, Contrastive, SemiSupervised)
├── models/      # registered models and example implementation
├── predictors/  # registered output post-processing utilities
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

## Hydra config files

Components can also be instantiated from YAML configs using Hydra. Example
configs live in the `config/` folder. They can be loaded with the helper
`instantiate_from_yaml`:

```python
from lightning_ml.core.utils.registry import instantiate_from_yaml

dataset = instantiate_from_yaml("config/dataset/labelled.yaml")
```

The framework stays intentionally small and flexible so you can plug in any PyTorch modules, metrics or Lightning `Trainer` configuration.

## License

This project is released under the terms of the MIT license. See [LICENSE](LICENSE) for details.


# Splitters
* CV is for validation, not deployment.
* Final model = retrained on all training data using the best config found via CV.
* So really we just split into
