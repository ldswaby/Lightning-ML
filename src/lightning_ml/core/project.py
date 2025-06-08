from typing import Optional

from pytorch_lightning import LightningDataModule, Trainer

from .learner import Learner


class Project:
    def __init__(
        self,
        learner: Learner,
        trainer: Optional[Trainer] = None,
    ):
        """Can pass in Trainer or None

        TODO: if predictor is None then make learner = Learner

        Args:
            learner (Learner): _description_
            predictor (PredictorMixin): _description_
            trainer (Optional[Trainer], optional): _description_. Defaults to None.
        """
        self.learner = learner
        self.trainer = trainer or Trainer()

    @property
    def data(self) -> LightningDataModule:
        return self.learner.data

    def train(self):
        self.trainer.fit(self.learner, datamodule=self.data)

    def validate(self):
        self.trainer.validate(self.learner, datamodule=self.data)

    def test(self):
        self.trainer.test(self.learner, datamodule=self.data)

    def predict(self, loaders=None):
        if loaders is None and self.data is not None:
            # predict internally stored dataloader
            loaders = self.data.predict_dataloader()
        # predict parsed dataloader
        return self.trainer.predict(self.learner, dataloaders=loaders)

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

    # @classmethod
    # def from_trainer_kwargs(
    #     cls, learner: Learner, predictor: PredictorMixin, *args, **kwargs
    # ):
    #     return cls(learner, predictor, Trainer(*args, **kwargs))
