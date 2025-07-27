# src/lightning_ml/core/utils/schema.py
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict

from lightning_ml.core.utils.enums import Registries
from lightning_ml.core.utils.registry import build, get_constructor


@dataclass
class RegistryConfig:
    """Configuration for a registry-backed component.

    Attributes:
        name: The registry key for the component.
        params: Initialization parameters for the component.
    """

    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the RegistryConfig to a dictionary.

        Returns:
            A dictionary with 'name' and 'params' keys.
        """
        return asdict(self)

    def instantiate(self, kind: Registries) -> Any:
        """Instantiate a component from the registry.

        Args:
            kind: Registry kind indicating which registry to use.

        Returns:
            An instance of the component.
        """
        return build(kind, self.name, **self.params)

    def get_constructor(self, kind: Registries) -> Callable[..., Any]:
        """Retrieve the constructor for a registry component without instantiating it.

        Args:
            kind: Registry kind indicating which registry to use.

        Returns:
            A class or function constructor for the component.
        """
        return get_constructor(kind, name=self.name)


@dataclass
class DataConfig:
    """Hierarchical config that first builds a Loader, then a Dataset via that Loader."""

    loader: RegistryConfig
    dataset: RegistryConfig

    def instantiate(self) -> Dict[str, Any]:
        # 1) build the loader instance
        loader_obj = self.loader.instantiate(Registries.LOADER)
        # 2) get the raw Dataset constructor
        ds_cls = self.dataset.get_constructor(Registries.DATASET)
        # 3) let loader turn that into a dataset
        return loader_obj.as_dataset(ds_cls, self.dataset.params)


@dataclass
class Config:

    data: DataConfig
    # model: RegistryConfig
    # optimizer: RegistryConfig
    seed: int = 42

    def instantiate(self) -> Dict[str, Any]:
        # return {key: getattr(self, key).instantiate(key) for key in Registries}
        return {
            key: getattr(self, key).instantiate(key) for key in ["loader", "dataset"]
        }


# conf/config.py
# from dataclasses import asdict, dataclass, field
# from pathlib import Path
# from typing import Any, ClassVar, Dict

# from lightning_ml.core.utils.enums import Registries
# from lightning_ml.core.utils.registry import build_from_cfg

# from .imports import requires

# @requires("hydra.utils", "omegaconf")
# def instantiate_from_yaml(cfg_path: str | Path) -> Any:
#     """Instantiate an object from a YAML config using Hydra.

#     The YAML file must contain a ``_target_`` key. When using the
#     :func:`build_from_cfg` helper this typically looks like::

#         _target_: lightning_ml.core.utils.registry.build_from_cfg
#         kind: dataset
#         name: LabelledDataset
#         params:
#             inputs: [1, 2]
#             targets: [3, 4]

#     Parameters
#     ----------
#     cfg_path : str or Path
#         Path to the YAML configuration file.

#     Returns
#     -------
#     Any
#         The instantiated object as returned by ``hydra.utils.instantiate``.
#     """

#     from hydra.utils import instantiate
#     from omegaconf import OmegaConf

#     cfg = OmegaConf.load(str(cfg_path))
#     return instantiate(cfg)


# @dataclass
# class _DictMixin:
#     """Adds a simple `to_dict` method"""

#     def to_dict(self) -> Dict[str, Any]:
#         return asdict(self)


# @dataclass(kw_only=True)
# class ConfigBase(_DictMixin):
#     name: str
#     params: Dict[str, Any] = field(default_factory=dict)

#     def instantiate(self) -> Any:
#         return build_from_cfg(config=self)


# @dataclass(kw_only=True)
# class DatasetConfig(ConfigBase):
#     kind: ClassVar[str] = Registries.DATASET


# @dataclass(kw_only=True)
# class LoaderConfig(ConfigBase):
#     kind: ClassVar[str] = Registries.LOADER


# @dataclass(kw_only=True)
# class ModelConfig(ConfigBase):
#     kind: ClassVar[str] = Registries.MODEL


# @dataclass
# class DataConfig(_DictMixin):

#     dataset: DatasetConfig
#     loader: LoaderConfig

#     def instantiate(self) -> Any:
#         return {
#             Registries.LOADER: getattr(self, Registries.LOADER).instantiate(),
#             Registries.DATASET: getattr(self, Registries.DATASET).instantiate(),
#         }


# # @dataclass(kw_only=True)
# # class OptimizerConfig:
# #     name: str
# #     params: Dict[str, Any] = field(default_factory=dict)


# @dataclass
# class Config:
#     # seed: int
#     dataset: DatasetConfig
#     model: ModelConfig
#     # optimizer: OptimizerConfig

#     def instantiate(self) -> Any:
#         return {k: v.instantiate() for k, v in self.to_dict().items()}


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
