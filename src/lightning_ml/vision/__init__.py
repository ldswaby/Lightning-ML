"""Vision modality package.

Importing this subpackage registers vision-specific components such as
Torchvision models and datasets if the dependency is available.
"""

from __future__ import annotations

from importlib import import_module

try:
    from ..utils.torchvision import register_torchvision

    register_torchvision()
except Exception:
    # torchvision not installed; silently skip registration
    pass

# Import submodules so any @register decorators run on import.
import_module("lightning_ml.vision.loaders")
import_module("lightning_ml.vision.classification.data.loaders")
import_module("lightning_ml.vision.detection.data.loaders")

__all__: list[str] = []
