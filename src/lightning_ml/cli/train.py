from __future__ import annotations

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torchmetrics import Accuracy

from lightning_ml.core.lightning_module import BaseLitModule
from lightning_ml.datamodules import DATAMODULE_REG
from lightning_ml.models import MODEL_REG


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    datamodule_cls = DATAMODULE_REG.get(cfg.datamodule.name)
    datamodule = datamodule_cls(**cfg.datamodule.params)

    model_cls = MODEL_REG.get(cfg.model.name)
    model = model_cls(**cfg.model.params)

    loss_fn = instantiate(cfg.loss)

    metrics = Accuracy()

    lit_model = BaseLitModule(
        model=model,
        loss_fn=loss_fn,
        optimizer_cfg=cfg.optimizer,
        scheduler_cfg=cfg.get("scheduler"),
        metrics=metrics,
    )

    trainer: Trainer = instantiate(cfg.trainer)
    trainer.fit(lit_model, datamodule=datamodule)


if __name__ == "__main__":
    main()
