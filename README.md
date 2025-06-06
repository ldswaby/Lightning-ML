# Lightning-ML
Private PyTorch-based machine learning library

# TODO:

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
