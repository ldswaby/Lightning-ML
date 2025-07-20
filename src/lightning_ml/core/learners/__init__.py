"""Learner registry and built-in learner classes."""

from __future__ import annotations

from ..utils.registry import Registry

LEARNER_REG = Registry("Learner")

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
