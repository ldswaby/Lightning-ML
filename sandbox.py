from pytorch_lightning import Trainer

from src.lightning_ml.core import Learner, PredictorMixin
from src.lightning_ml.learners import Supervised
from src.lightning_ml.predictors import Classification
from src.lightning_ml.utils import bind_classes

# #############
# model = NotImplemented
# optimizer = NotImplemented
# data = NotImplemented
# criterion = NotImplemented
# metrics = NotImplemented
# scheduler = NotImplemented
# #############

student = Supervised.with_predictor(
    Classification,
    model=model,
    optimizer=optimizer,
    data=data,
    criterion=crtierion,
    metrics=metrics,
    scheduler=None,
)

student = bind_classes(Supervised, Classification)(
    model=model,
    optimizer=optimizer,
    data=data,
    criterion=crtierion,
    metrics=metrics,
    scheduler=None,  # TODO
)

# QUESTION: I don't like either of the above.
# Why can't I define:
# 1. Learner (with optionally combined PredictorMixin to make a 'student' - learner + optinal predictor)
# 3. Trainer (pytorch_lighting.Trainer)
# 4. Project object which combines all the above
