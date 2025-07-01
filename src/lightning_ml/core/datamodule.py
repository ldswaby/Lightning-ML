from typing import Any, Dict, Optional

import pytorch_lightning as pl
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset


class DataModule(pl.LightningDataModule):
    """
    LightningDataModule supporting both holdout and k-fold cross-validation.

    This DataModule splits a full dataset into training and validation subsets
    using either a simple holdout split (when `k_folds=1`) or K-fold
    cross-validation (when `k_folds>1`).
    """

    def __init__(
        self,
        full_dataset: Dataset,
        k_folds: int = 1,
        fold_idx: int = 0,
        test_size: float = 0.2,
        shuffle: bool = True,
        random_state: int = 42,
        **dataloader_kwargs: Any,
    ) -> None:
        """
        Initializes the DataModule.

        Args:
            full_dataset: Dataset to split.
            k_folds: Number of folds for cross-validation. If 1, holdout split is used.
            fold_idx: Index of the validation fold (0-based).
            test_size: Proportion of the dataset to use as validation in holdout mode.
            shuffle: Whether to shuffle before splitting.
            random_state: Seed for reproducibility.
            **dataloader_kwargs: Additional arguments for DataLoader.
        """
        super().__init__()
        self.full_dataset: Dataset = full_dataset
        self.k_folds: int = k_folds
        self.fold_idx: int = fold_idx
        self.test_size: float = test_size
        self.shuffle: bool = shuffle
        self.random_state: int = random_state
        self.dataloader_kwargs: Dict[str, Any] = dataloader_kwargs
        self.train_dataset: Optional[Subset] = None
        self.val_dataset: Optional[Subset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Splits the dataset into training and validation subsets.

        Args:
            stage: Stage name (unused).
        """
        total_indices = list(range(len(self.full_dataset)))
        if self.k_folds > 1:
            kf = KFold(
                n_splits=self.k_folds,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
            splits = list(kf.split(total_indices))
            train_idx, val_idx = splits[self.fold_idx]
        else:
            train_idx, val_idx = train_test_split(
                total_indices,
                test_size=self.test_size,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
        self.train_dataset = Subset(self.full_dataset, train_idx)
        self.val_dataset = Subset(self.full_dataset, val_idx)

    def train_dataloader(self) -> DataLoader:
        """Returns DataLoader for the training dataset."""
        return DataLoader(
            self.train_dataset,
            shuffle=self.shuffle,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns DataLoader for the validation dataset."""
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            **self.dataloader_kwargs,
        )


### EXAMPLE

# import pytorch_lightning as pl
# from torch.utils.data      import DataLoader, Subset
# from sklearn.model_selection import KFold

# # --- 1) Your LightningModule ---
# class MyLightningModule(pl.LightningModule):
#     def __init__(self, lr=1e-3):
#         super().__init__()
#         self.save_hyperparameters()
#         # define model, loss, etc.
#         self.model = ...
#         self.criterion = ...

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         preds = self(x)
#         loss = self.criterion(preds, y)
#         self.log('train_loss', loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         preds = self(x)
#         loss = self.criterion(preds, y)
#         acc  = (preds.argmax(dim=-1) == y).float().mean()
#         self.log('val_loss', loss, prog_bar=True)
#         self.log('val_acc',  acc,  prog_bar=True)

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# # --- 2) Full dataset (e.g. torchvision) ---
# full_dataset = torchvision.datasets.CIFAR10(...)

# # --- 3) Cross-validation loop ---
# k_folds = 5
# kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# metrics_per_fold = []
# for fold_idx, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
#     print(f'=== Fold {fold_idx} ===')

#     train_ds = Subset(full_dataset, train_idx)
#     val_ds   = Subset(full_dataset, val_idx)

#     train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
#     val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4)

#     model   = MyLightningModule(lr=1e-3)
#     trainer = pl.Trainer(
#         max_epochs=10,
#         gpus=1,
#         logger=pl.loggers.TensorBoardLogger("tb_logs", name=f"fold_{fold_idx}")
#     )

#     trainer.fit(model, train_loader, val_loader)
#     metrics_per_fold.append(trainer.callback_metrics)

# # --- 4) Compute average across folds ---
# avg_metrics = {}
# for key in metrics_per_fold[0]:
#     values = [m[key].item() for m in metrics_per_fold]
#     avg_metrics[key] = sum(values) / k_folds

# print("Average cross-val metrics:", avg_metrics)
