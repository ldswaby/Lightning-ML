from pytorch_lightning import Trainer

from src.lightning_ml.core import Learner, PredictorMixin
from src.lightning_ml.learners import Supervised
from src.lightning_ml.predictors import Classification

student = Supervised.with_predictor(Classification)


# def bind(learner: type[Learner], predictor: type[PredictorMixin]):
#     """
#     TODO: perhaps this should be a `Project` class itself, so it can wrap all the obove with additional logic such as running Task.fit() under Project.train() etc, and use all the correct internal variables


#     Dynamically create a new LightningModule that inherits training logic
#     from `problem_cls` and inference logic from `task_mixin`.
#     """
#     name = f"{learner.__name__}{predictor.__name__}"
#     return type(name, (predictor, learner), {})


# SupervisedClassification = bind(Supervised, Classification)

# #############
# model = NotImplemented
# optimizer = NotImplemented
# data = NotImplemented
# criterion = NotImplemented
# metrics = NotImplemented
# scheduler = NotImplemented

# #############

# task = SupervisedClassification(
#     model=model,
#     optimizer=optimizer,
#     data=data,
#     criterion=crtierion,
#     metrics=metrics,
#     scheduler=None,  # TODO
# )

# trainer = Trainer(limit_train_batches=100, max_epochs=1)
# trainer.fit(model=model, train_dataloaders=data.train)

# # should it not then be task.train()? I.e. should one of my objects not already contain a task object?
