"""
Utility helpers for dependency management and dynamic imports used throughout the
:pyproj:`lightning_ml` code‑base.

The main entry‑point is :pyfunc:`requires`, a decorator factory that guards a callable
so it executes only when its optional dependencies are present. When a requirement is
missing, the wrapped callable raises a :class:`ModuleNotFoundError` with an actionable
*pip install* hint instead of failing at import‑time.

Example
-------
    ```python
    @requires("torch", "torchvision")
    def forward():
        ...
    ```
If either *torch* or *torchvision* are not installed, calling *forward* raises a clear
error indicating which packages need to be installed.

The module also exposes:

* :pyfunc:`example_requires` – convenience wrapper for sample scripts.
* :pyfunc:`import_from_str` – lightweight utility to import a symbol from its dotted
  path (e.g. ``"torch.nn.Linear"``).

All public symbols are re‑exported via :pydata:`__all__`.
"""

import contextlib
import functools
import importlib
import operator
import types
import sys
from typing import Any, Callable, List, Tuple, TypeVar, Union

from lightning_utilities.core.imports import compare_version, module_available
from pkg_resources import DistributionNotFound

__all__: list[str] = [
    "requires",
    "example_requires",
    "import_from_str",
]


# Generic type used by decorators
F = TypeVar("F", bound=Callable[..., Any])


try:
    from packaging.version import Version
except (ModuleNotFoundError, DistributionNotFound):
    Version = None


def _safe_module_available(name: str) -> bool:
    """Return True if the given module can be imported.

    This wrapper mirrors :func:`lightning_utilities.core.imports.module_available`
    but gracefully handles modules already present in ``sys.modules`` that may
    not have a proper ``__spec__`` (as is the case in the test suite).
    """
    if name in sys.modules:
        return True
    try:
        return module_available(name)
    except Exception:
        return False

_TORCH_AVAILABLE = _safe_module_available("torch")
_PL_AVAILABLE = _safe_module_available("pytorch_lightning")
_BOLTS_AVAILABLE = _safe_module_available("pl_bolts") and compare_version(
    "torch", operator.lt, "1.9.0"
)
_PANDAS_AVAILABLE = _safe_module_available("pandas")
_SKLEARN_AVAILABLE = _safe_module_available("sklearn")
_PYTORCHTABULAR_AVAILABLE = _safe_module_available("pytorch_tabular")
_FORECASTING_AVAILABLE = _safe_module_available("pytorch_forecasting")
_KORNIA_AVAILABLE = _safe_module_available("kornia")
_COCO_AVAILABLE = _safe_module_available("pycocotools")
_TIMM_AVAILABLE = _safe_module_available("timm")
_TORCHVISION_AVAILABLE = _safe_module_available("torchvision")
_PYTORCHVIDEO_AVAILABLE = _safe_module_available("pytorchvideo")
_MATPLOTLIB_AVAILABLE = _safe_module_available("matplotlib")
_TRANSFORMERS_AVAILABLE = _safe_module_available("transformers")
_PYSTICHE_AVAILABLE = _safe_module_available("pystiche")
with contextlib.suppress(ConnectionError):
    _FIFTYONE_AVAILABLE = _safe_module_available("fiftyone")
_FASTAPI_AVAILABLE = _safe_module_available("fastapi")
_PYDANTIC_AVAILABLE = _safe_module_available("pydantic")
_GRAPHVIZ_AVAILABLE = _safe_module_available("graphviz")
_CYTOOLZ_AVAILABLE = _safe_module_available("cytoolz")
_UVICORN_AVAILABLE = _safe_module_available("uvicorn")
_PIL_AVAILABLE = _safe_module_available("PIL")
_OPEN3D_AVAILABLE = _safe_module_available("open3d")
_SEGMENTATION_MODELS_AVAILABLE = _safe_module_available("segmentation_models_pytorch")
_FASTFACE_AVAILABLE = _safe_module_available("fastface") and compare_version(
    "pytorch_lightning", operator.lt, "1.5.0"
)
_LIBROSA_AVAILABLE = _safe_module_available("librosa")
_TORCH_SCATTER_AVAILABLE = _safe_module_available("torch_scatter")
_TORCH_SPARSE_AVAILABLE = _safe_module_available("torch_sparse")
_TORCH_GEOMETRIC_AVAILABLE = _safe_module_available("torch_geometric")
_NETWORKX_AVAILABLE = _safe_module_available("networkx")
_TORCHAUDIO_AVAILABLE = _safe_module_available("torchaudio")
_SENTENCEPIECE_AVAILABLE = _safe_module_available("sentencepiece")
_DATASETS_AVAILABLE = _safe_module_available("datasets")
_TM_TEXT_AVAILABLE: bool = _safe_module_available("torchmetrics.text")
_ICEVISION_AVAILABLE = _safe_module_available("icevision")
_ICEDATA_AVAILABLE = _safe_module_available("icedata")
_LEARN2LEARN_AVAILABLE = _safe_module_available("learn2learn") and compare_version(
    "learn2learn", operator.ge, "0.1.6"
)
_TORCH_ORT_AVAILABLE = _safe_module_available("torch_ort")
_VISSL_AVAILABLE = _safe_module_available("vissl") and _safe_module_available("classy_vision")
_ALBUMENTATIONS_AVAILABLE = _safe_module_available("albumentations")
_BAAL_AVAILABLE = _safe_module_available("baal")
_TORCH_OPTIMIZER_AVAILABLE = _safe_module_available("torch_optimizer")
_SENTENCE_TRANSFORMERS_AVAILABLE = _safe_module_available("sentence_transformers")
_DEEPSPEED_AVAILABLE = _safe_module_available("deepspeed")
_EFFDET_AVAILABLE = _safe_module_available("effdet")


if _PIL_AVAILABLE:
    from PIL import Image  # noqa: F401
else:

    class Image:
        Image = object


if Version:
    _TORCHVISION_GREATER_EQUAL_0_9 = compare_version(
        "torchvision", operator.ge, "0.9.0"
    )
    _PL_GREATER_EQUAL_1_8_0 = compare_version("pytorch_lightning", operator.ge, "1.8.0")
    _PANDAS_GREATER_EQUAL_1_3_0 = compare_version("pandas", operator.ge, "1.3.0")
    _ICEVISION_GREATER_EQUAL_0_11_0 = compare_version(
        "icevision", operator.ge, "0.11.0"
    )
    _TM_GREATER_EQUAL_0_10_0 = compare_version("torchmetrics", operator.ge, "0.10.0")
    _BAAL_GREATER_EQUAL_1_5_2 = compare_version("baal", operator.ge, "1.5.2")
    _TRANSFORMERS_GREATER_EQUAL_4_0 = compare_version(
        "transformers", operator.ge, "4.0.0"
    )

_TOPIC_TEXT_AVAILABLE = all(
    [
        _TRANSFORMERS_AVAILABLE,
        _SENTENCEPIECE_AVAILABLE,
        _DATASETS_AVAILABLE,
        _TM_TEXT_AVAILABLE,
        _SENTENCE_TRANSFORMERS_AVAILABLE,
    ]
)
_TOPIC_TABULAR_AVAILABLE = all(
    [_PANDAS_AVAILABLE, _FORECASTING_AVAILABLE, _PYTORCHTABULAR_AVAILABLE]
)
_TOPIC_VIDEO_AVAILABLE = all(
    [_TORCHVISION_AVAILABLE, _PIL_AVAILABLE, _PYTORCHVIDEO_AVAILABLE, _KORNIA_AVAILABLE]
)
_TOPIC_IMAGE_AVAILABLE = all(
    [
        _TORCHVISION_AVAILABLE,
        _TIMM_AVAILABLE,
        _PIL_AVAILABLE,
        _ALBUMENTATIONS_AVAILABLE,
        _PYSTICHE_AVAILABLE,
    ]
)
_TOPIC_SERVE_AVAILABLE = all(
    [_FASTAPI_AVAILABLE, _PYDANTIC_AVAILABLE, _CYTOOLZ_AVAILABLE, _UVICORN_AVAILABLE]
)
_TOPIC_POINTCLOUD_AVAILABLE = all([_OPEN3D_AVAILABLE, _TORCHVISION_AVAILABLE])
_TOPIC_AUDIO_AVAILABLE = all(
    [
        _TORCHAUDIO_AVAILABLE,
        _TORCHVISION_AVAILABLE,
        _LIBROSA_AVAILABLE,
        _TRANSFORMERS_AVAILABLE,
    ]
)
_TOPIC_GRAPH_AVAILABLE = all(
    [
        _TORCH_SCATTER_AVAILABLE,
        _TORCH_SPARSE_AVAILABLE,
        _TORCH_GEOMETRIC_AVAILABLE,
        _NETWORKX_AVAILABLE,
    ]
)
_TOPIC_CORE_AVAILABLE = all(
    [_TOPIC_IMAGE_AVAILABLE, _TOPIC_TABULAR_AVAILABLE, _TOPIC_TEXT_AVAILABLE]
)

_EXTRAS_AVAILABLE = {
    "image": _TOPIC_IMAGE_AVAILABLE,
    "tabular": _TOPIC_TABULAR_AVAILABLE,
    "text": _TOPIC_TEXT_AVAILABLE,
    "video": _TOPIC_VIDEO_AVAILABLE,
    "pointcloud": _TOPIC_POINTCLOUD_AVAILABLE,
    "serve": _TOPIC_SERVE_AVAILABLE,
    "audio": _TOPIC_AUDIO_AVAILABLE,
    "graph": _TOPIC_GRAPH_AVAILABLE,
}


def requires(*module_paths: str | tuple[bool, str]) -> Callable[[F], F]:
    """Decorator factory to enforce optional runtime dependencies.

    The decorator checks that all requested modules *or* extras are available
    before allowing the wrapped callable to execute. Availability is evaluated
    lazily at **call** time, not at **import** time, so that downstream modules
    can be imported regardless of the user's local environment.

    Args:
        *module_paths: Each element is either

            * **str** – a module name tested with
              :func:`lightning_utilities.core.imports.module_available` **or**
              a key of :pydata:`_EXTRAS_AVAILABLE` (e.g. ``"image"``, ``"text"``).
            * **tuple[bool, str]** – a pre‑computed availability flag and human
              readable module name.

    Returns:
        Callable[[~F], ~F]: A decorator preserving the signature of *func*. If
        dependencies are satisfied, it immediately invokes the function;
        otherwise it raises :class:`ModuleNotFoundError` with installation
        instructions.

    Raises:
        ModuleNotFoundError: If at least one required dependency is missing.

    Example:
        ```python
        @_requires("torch", "pandas", (has_gpu, "CUDA Toolkit"))
        def heavy_lifting():
            ...
        ```
    """

    def decorator(func):
        available = True
        extras = []
        modules = []
        for module_path in module_paths:
            if isinstance(module_path, str):
                if module_path in _EXTRAS_AVAILABLE:
                    extras.append(module_path)
                    if not _EXTRAS_AVAILABLE[module_path]:
                        available = False
                else:
                    modules.append(module_path)
                    if not _safe_module_available(module_path):
                        available = False
            else:
                available, module_path = module_path
                modules.append(module_path)

        if not available:
            modules = [f"'{module}'" for module in modules]

            if extras:
                modules.append(f"'lightning-flash[{','.join(extras)}]'")

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                raise ModuleNotFoundError(
                    f"Required dependencies not available. Please run: pip install {' '.join(modules)}"
                )

            return wrapper
        return func

    return decorator


def example_requires(module_paths: str | list[str]) -> None:
    """Eagerly validate example‑level dependencies.

    This convenience helper is intended for sample scripts and tutorials where
    missing dependencies should cause an early, explicit failure at import
    time (rather than at function call time).

    Args:
        module_paths: A single module name or list of modules/extras to test.

    Raises:
        ModuleNotFoundError: If *any* of the requested dependencies are absent.
    """
    return requires(module_paths)(lambda: None)()


def import_from_str(qual: str):
    """Import an attribute from its fully‑qualified dotted path.

    Args:
        qual: A dotted path like ``"torch.nn.Linear"``.

    Returns:
        The imported attribute.

    Raises:
        ValueError: If *qual* does not contain a ``"."`` separator.
        ModuleNotFoundError: If the target module cannot be imported.
        AttributeError: If the attribute does not exist in the module.

    Example:
        >>> Linear = import_from_str("torch.nn.Linear")
        >>> isinstance(Linear, type)
        True
    """
    mod, cls = qual.rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)
