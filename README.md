# Lightning-ML
Private PyTorch-based machine learning library

# TODO:

* There is currently confusion in that you want the set of problems to be optionally wrapped by the set of tasks that will add processing steps to the model outputs in order to produce predictions (i.e. for testing which may not be needed for training). However, currently, you are using `predict_step` all the way through so that the logic written there will run in both training and testing
* See TODOs, but the task is the loss etc, as in how the model is trained, but the downstream task/inference is different and may require additional logic. E.g. contrastive model used for classification? This would happen in a hook somehow before this return value, also defined by user (perhaps configurable). Maybe, there's the means by which the model is trained Contrastive etc, then the inference Task, regression, classification etc
* In the same way the tasks are modular, so must be the datasets and models so they can all be mapped together to ensure the outputs of one don't break the next
* Implement `hydra` config parsing
* For data, use `pl.LightningDataModule`
* For model just use `nn.Module`
* For Task use a `pl.LightningModule` which organizes code into 6 sections:
    * Initialization (`__init__` and `setup()`). Model will be selected here, based on config.
    * Train Loop (`training_step()`)
    * Validation Loop (`validation_step()`)
    * Test Loop (`test_step()`)
    * Prediction Loop (`predict_step()`)
    * Optimizers and LR Schedulers (`configure_optimizers()`)

# Components:
* Dataset/Dataloader
* Model
* Loss / Metrics
* Optimizer
* Scheduler

# Conceptualization
the typical ML workflow

A learning problem involves a learner (model), learning material (data), a learning method (optimizer), and some form of feedback (loss).
In this object we simply define the way inputs flow through each of these compnents: Data -> Model -> Loss

Code maps learner (lightning module model trainer) to a predictor mixin, that will determine how model outputs are postprocesses

Note: Call a learner and predictor together a `Student`
