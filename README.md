# Lightning-ML
Private PyTorch-based machine learning library

# TODO:
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
