"""Compatibility module exposing data utilities under :mod:`lightning_ml.data`."""

from ..core import data as _core_data
from ..core.data import *  # noqa: F401,F403

__all__ = getattr(_core_data, "__all__", [])
