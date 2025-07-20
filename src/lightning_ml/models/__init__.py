"""Model registry and built-in models."""

from __future__ import annotations

from ..core.utils.registry import Registry

MODEL_REG = Registry("Model")

from .example import *  # noqa: F401,F403

__all__ = ["MODEL_REG", "MyCustomModel"]
