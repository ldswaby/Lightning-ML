"""Lightning-ML base package."""

from .datasets import DATAMODULE_REG
from .models import MODEL_REG

__all__ = ["MODEL_REG", "DATAMODULE_REG"]
