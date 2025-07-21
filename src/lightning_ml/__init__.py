"""Lightning-ML top-level package.

Importing :mod:`lightning_ml` initialises the core registries so the base
components are always registered.  Modality packages (e.g.
:mod:`lightning_ml.vision`) register their components only when imported.
"""

from __future__ import annotations

from importlib import import_module

# Import modules that create the core registries and register their built-in
# components.  Import order doesn't matter as each registry is independent.
import_module("lightning_ml.core.data.datasets")
try:
    import_module("lightning_ml.core.data.loaders")
except Exception:  # pragma: no cover - optional deps may be missing
    pass
import_module("lightning_ml.core.learners")
import_module("lightning_ml.core.predictors")

__all__: list[str] = []
