"""Learner registry and built-in learner classes."""

from __future__ import annotations

from ..utils.enums import Registries
from ..utils.registry import get_registry

LEARNER_REG = get_registry(Registries.LEARNER)

from .contrastive import Contrastive  # noqa: F401
from .supervised import Supervised  # noqa: F401
from .unsupervised import Unsupervised  # noqa: F401
from .semi_supervised import SemiSupervised  # noqa: F401

__all__ = [
    "LEARNER_REG",
    "Supervised",
    "Unsupervised",
    "SemiSupervised",
    "Contrastive",
]
