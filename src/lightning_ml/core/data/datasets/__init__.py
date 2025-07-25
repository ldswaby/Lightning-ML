"""Dataset registry and built-in dataset classes."""

from __future__ import annotations

from lightning_ml.core.utils.enums import Registries
from lightning_ml.core.utils.registry import get_registry

# Define the registry before importing submodules that register with it to
# avoid import cycles during package initialisation.
REGISTRY = get_registry(Registries.DATASET)

# Import dataset implementations which will register themselves with
# ``DATASET_REG`` on import.
from .contrastive import *
from .labelled import *
from .unlabelled import *
from .numpy import *
