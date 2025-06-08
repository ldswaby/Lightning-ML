from typing import Any, Dict

import torch

from lightning_ml.learners.abstract import Supervised, Unsupervised


class SupContrastive(Supervised):

    def step(self, batch) -> Dict[str, Any]:
        (v1, v2), y = batch
        z1, z2 = self(v1), self(v2)
        loss = self.criterion(torch.stack([z1, z2], dim=1), y)
        return {
            "loss": loss
        }  # TODO the task is the loss etc, as in how the model is trained, but the downstream task/inference is different and may require additional logic. E.g. contrastive model used for classification? This would happen in a hook somehow before this return value, also defined by user (perhaps configurable). Maybe, should be split as follows: there's the means by which the model is trained (sup, unsup, contrastive etc, then the inference Task, regression, classification etc


class SelfSupContrastive(Unsupervised):

    def step(self, batch):
        v1, v2 = batch
        z1, z2 = self(v1), self(v2)
        loss = self.criterion(z1, z2)
        return {"loss": loss}
