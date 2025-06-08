from typing import Optional

from pytorch_lightning import LightningDataModule, Trainer

from .learner import Learner


class Project:
    def __init__(
        self,
        student: Learner,
        trainer: Optional[Trainer] = None,
    ):
        """Can pass in Trainer or None

        TODO: if predictor is None then make student = Learner

        Args:
            learner (Learner): _description_
            trainer (Optional[Trainer], optional): _description_. Defaults to None.
        """
        self.student = student
        self.trainer = trainer or Trainer()

    @property
    def data(self) -> LightningDataModule:
        return self.student.data

    def train(self):
        self.trainer.fit(self.student, datamodule=self.data)

    def validate(self):
        self.trainer.validate(self.student, datamodule=self.data)

    def test(self):
        self.trainer.test(self.student, datamodule=self.data)

    def predict(self, loaders=None):
        if loaders is None and self.data is not None:
            # predict internally stored dataloader
            loaders = self.data.predict_dataloader()
        # predict parsed dataloader
        return self.trainer.predict(self.student, dataloaders=loaders)

    def __getattr__(self, item):
        """Delegate other Trainer attrs

        Args:
            item (_type_): _description_

        Raises:
            AttributeError: _description_

        Returns:
            _type_: _description_
        """
        if hasattr(self.trainer, item):
            return getattr(self.trainer, item)
        raise AttributeError(item)

