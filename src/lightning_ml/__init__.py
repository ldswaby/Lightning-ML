"""Lightning-ML base package."""

from .models import MODEL_REG
from .datamodules import DATAMODULE_REG

__all__ = ["MODEL_REG", "DATAMODULE_REG"]
