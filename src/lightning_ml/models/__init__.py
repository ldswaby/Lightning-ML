"""Model registry and built-in models."""

from __future__ import annotations

from ..core.utils.enums import Registries
from ..core.utils.registry import get_registry

MODEL_REG = get_registry(Registries.MODEL)

# Example models can register themselves here
# from .example import *  # noqa: F401,F403

__all__ = ["MODEL_REG"]
