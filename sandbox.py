from pytorch_lightning import Trainer

from src.lightning_ml.core import Project
from src.lightning_ml.learners import Supervised
from src.lightning_ml.predictors import Classification

# #############
# model = NotImplemented
# optimizer = NotImplemented
# data = NotImplemented
# criterion = NotImplemented
# metrics = NotImplemented
# scheduler = NotImplemented
# #############

# Create a Learner and optionally combine it with a Predictor using ``Project``.
project = Project(
    Supervised,
    predictor=Classification,
    model=model,
    optimizer=optimizer,
    data=data,
    criterion=crtierion,
    metrics=metrics,
    scheduler=None,  # TODO
)

# The ``Project`` instance exposes Trainer-like methods for convenience
# and can be used as the single entry point for training or inference.
