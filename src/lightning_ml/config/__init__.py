"""Hydra integration utilities for Lightning-ML."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


from ..core.utils.enums import Registries
from ..core.utils.registry import get_registry

__all__ = ["RegistryConfig", "instantiate_from_registry"]


def instantiate_from_registry(
    registry: Registries | str,
    name: str,
    params: Dict[str, Any] | None = None,
) -> Any:
    """Instantiate an object from one of the global registries.

    Parameters
    ----------
    registry: :class:`Registries` or ``str``
        The registry to look up the object in.
    name: str
        Key of the object in the registry.
    params: dict, optional
        Keyword arguments passed to the object's constructor.

    Returns
    -------
    Any
        Instantiated object from the registry.
    """
    if isinstance(registry, str):
        registry = Registries(registry)
    cls = get_registry(registry).get(name)
    return cls(**(params or {}))


@dataclass
class RegistryConfig:
    """Hydra configuration for objects stored in a registry."""

    registry: Registries | str
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    _target_: str = field(
        init=False, default="lightning_ml.config.instantiate_from_registry"
    )



@dataclass
class TrainConfig:
    """Top-level Hydra configuration for training."""

    datamodule: RegistryConfig
    model: RegistryConfig
    learner: RegistryConfig
    trainer: Dict[str, Any] = field(default_factory=lambda: {"_target_": "pytorch_lightning.Trainer"})
