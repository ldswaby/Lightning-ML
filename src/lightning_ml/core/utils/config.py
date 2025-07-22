# conf/config.py
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict

from lightning_ml.core.utils.registry import build_from_cfg
from lightning_ml.core.utils.enums import Registries

from .imports import requires


@requires("hydra.utils", "omegaconf")
def instantiate_from_yaml(cfg_path: str | Path) -> Any:
    """Instantiate an object from a YAML config using Hydra.

    The YAML file must contain a ``_target_`` key. When using the
    :func:`build_from_cfg` helper this typically looks like::

        _target_: lightning_ml.core.utils.registry.build_from_cfg
        kind: dataset
        name: LabelledDataset
        params:
            inputs: [1, 2]
            targets: [3, 4]

    Parameters
    ----------
    cfg_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    Any
        The instantiated object as returned by ``hydra.utils.instantiate``.
    """

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(cfg_path))
    return instantiate(cfg)



@dataclass
class ConfigBase:

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def instantiate(self) -> Any:
        return build_from_cfg(config=self)


@dataclass
class DatasetConfig(ConfigBase):
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    kind: str = Registries.DATASET


@dataclass
class ModelConfig(ConfigBase):
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    kind: str = Registries.MODEL


# @dataclass
# class OptimizerConfig:
#     name: str
#     params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    # optimizer: OptimizerConfig


# main.py
# import hydra
# from hydra.core.config_store import ConfigStore
# from omegaconf import DictConfig

# from conf.config import Config

# cs = ConfigStore.instance()
# cs.store(name="config", node=Config)


# @hydra.main(config_name="config", config_path="conf")
# def main(cfg: Config) -> None:
#     """Entry point for application."""
#     print(f"Running with seed={cfg.seed}")
#     print(f"Model: {cfg.model.name}, lr={cfg.model.lr}")
#     print(f"Optimizer: {cfg.optimizer.type}, wd={cfg.optimizer.weight_decay}")


# if __name__ == "__main__":
#     main()


# conf/
# ├── config.yaml       # default “root” config
# ├── model/
# │   ├── resnet.yaml
# │   └── vgg.yaml
# └── optimizer/
#     ├── adam.yaml
#     └── sgd.yaml


# defaults:
#   - model: resnet
#   - optimizer: adam

# seed: 42


# # Hydra will merge this under `cfg.model`
# name: "resnet50"
# lr: 0.001
# dropout: 0.1


# # Change model to vgg, override learning rate
# python main.py model=vgg model.lr=0.0005

# # Override root-level
# python main.py seed=2025
