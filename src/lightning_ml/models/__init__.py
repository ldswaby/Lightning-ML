"""Model registry and built-in models."""

from __future__ import annotations

from ..core.utils.enums import Registries
from ..core.utils.registry import get_registry

MODEL_REG = get_registry(Registries.MODEL)

__all__ = ["MODEL_REG"]
