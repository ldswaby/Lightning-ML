# configs/datamodule.py
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class LoaderConfig:
    name: str  # key in loader registry
    kwargs: Dict[str, Any] = field(default_factory=dict)


# @dataclass
# class DataModuleConfig:
#     dataset: str  # key in DATASET_REG
#     dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
#     dataloader_kwargs: Dict[str, Any] = field(default_factory=dict)


# # configs/model.py
# @dataclass
# class ModelConfig:
#     model: str  # key in MODEL_REG
#     model_kwargs: Dict[str, Any] = field(default_factory=dict)


# # configs/learner.py
# @dataclass
# class LearnerConfig:
#     learner: str  # key in LEARNER_REG
#     learner_kwargs: Dict[str, Any] = field(default_factory=dict)
