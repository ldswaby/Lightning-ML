"""High level interface for training and inference."""

from typing import Optional

from pytorch_lightning import LightningDataModule, Trainer

from .learner import Learner


class Project:
    """Orchestrates a :class:`Learner` and a :class:`~pytorch_lightning.Trainer`."""

    def __init__(self, learner: Learner, trainer: Optional[Trainer] = None) -> None:
        """Create a new project.

        Parameters
        ----------
        learner : Learner
            Learner instance encapsulating the model and training logic.
        trainer : Trainer, optional
            PyTorch Lightning trainer used to run the experiment. If ``None`` a
            default ``Trainer()`` is created.
        """
        self.learner = learner
        self.trainer = trainer or Trainer()

    # ------------------------------------------------------------------
    @property
    def data(self) -> Optional[LightningDataModule]:
        """Convenience access to the learner's data_module."""
        return self.learner.data_module

    # ------------------------------------------------------------------
    # Convenience wrappers around ``Trainer`` methods
    # ------------------------------------------------------------------
    def train(self) -> None:
        """Fit the learner."""
        self.trainer.fit(self.learner, datamodule=self.data)

    def validate(self) -> None:
        """Run validation."""
        self.trainer.validate(self.learner, datamodule=self.data)

    def test(self) -> None:
        """Run testing."""
        self.trainer.test(self.learner, datamodule=self.data)

    def predict(self, loaders=None):
        """Run prediction using the learner's predictor."""
        if loaders is None and self.data is not None:
            loaders = self.data.predict_dataloader()
        return self.trainer.predict(self.learner, dataloaders=loaders)

    # ------------------------------------------------------------------
    def __getattr__(self, item):
        """Delegate attribute access to the underlying trainer."""
        if hasattr(self.trainer, item):
            return getattr(self.trainer, item)
        raise AttributeError(item)
