"""Generic registry for components."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, TypeVar

from .enums import Registries
from .imports import import_from_str, requires

__all__ = [
    "Registry",
    "REGISTRIES",
    "get_registry",
    "register",
    "build",
    "build_from_cfg",
    "instantiate_from_yaml",
]

T = TypeVar("T")


class Registry(dict):
    """Dictionary-like registry of callables or classes."""

    def __init__(self, lib: str) -> None:
        """Initialize a new registry.

        Args:
            lib: A human‑readable label describing the type of component being registered.
        """
        super().__init__()
        self._lib = lib

    def register(self, name: str | None = None) -> Callable[[T], T]:
        """Return a decorator that registers a class or callable.

        Args:
            name: Optional explicit name under which to register the object. If omitted, the object's ``__name__`` is used.

        Returns:
            Callable[[T], T]: The decorator that registers the object.

        Examples:
            >>> reg = Registry("demo")
            >>> @reg.register()
            ... class MyCls:
            ...     pass
            >>> reg.list_keys()
            ['MyCls']

        Raises:
            KeyError: If the given ``name`` is already present in the registry.
        """

        def decorator(cls: T) -> T:
            key = name or cls.__name__
            if key in self:
                raise KeyError(f"{self._lib} '{key}' is already registered")
            self[key] = cls
            return cls

        return decorator

    def get(self, name: str) -> T:
        """Retrieve a registered component.

        Args:
            name: Name of the component to retrieve.

        Returns:
            T: The requested component.

        Examples:
            >>> reg = Registry("demo")
            >>> @reg.register()
            ... class YourCls:
            ...     pass
            >>> reg.get("YourCls") is YourCls
            True

        Raises:
            KeyError: If ``name`` is not found in the registry.
        """
        if name not in self:
            raise KeyError(f"{self._lib} '{name}' not found in registry")
        return super().__getitem__(name)

    def list_keys(self) -> list[str]:  # pragma: no cover - trivial
        """Return the list of registered component names.

        Examples:
            >>> reg = Registry("demo")
            >>> @reg.register()
            ... class Foo:
            ...     pass
            >>> reg.list_keys()
            ['Foo']
        """
        return list(self.keys())


# Global enum-based registries


# Internal mapping of string keys to Registry instances
REGISTRIES: dict[str, Registry] = {}


def _to_key(kind: Any) -> str:
    """Return a registry key as a string.

    Args:
        kind: Either an enum member with a ``value`` attribute or a plain
            string.

    Returns:
        str: The canonical string key.

    Examples:
        >>> from enum import Enum
        >>> class Kind(Enum):
        ...     FOO = "foo"
        >>> _to_key("bar")
        'bar'
        >>> _to_key(Kind.FOO)
        'foo'
    """
    return kind.value if hasattr(kind, "value") else kind


def get_registry(kind: Any) -> Registry:
    """Get or create the :class:`Registry` associated with ``kind``.

    Args:
        kind: Enum value or string that identifies the registry.

    Returns:
        Registry: The corresponding registry instance.

    Examples:
        >>> from lightning_ml.core.utils.registry import get_registry
        >>> reg1 = get_registry("demo")
        >>> reg2 = get_registry("demo")
        >>> reg1 is reg2
        True
    """
    key = _to_key(kind)
    return REGISTRIES.setdefault(key, Registry(key))


def register(enum_val: Any, name: str | None = None) -> Callable[[T], T]:
    """Return a decorator that registers a class or callable.

    Args:
        enum_val: Registry bucket identified either by an enum member or a
            string such as ``"model"``.
        name: Optional explicit name to register the object under. If
            ``None``, the object’s ``__name__`` is used.

    Returns:
        Callable[[T], T]: A decorator that registers the object into the
        target registry.

    Examples:
        >>> from lightning_ml.core.utils.registry import register, get_registry
        >>> @register("model")
        ... class TinyNet:
        ...     pass
        >>> get_registry("model").get("TinyNet") is TinyNet
        True
    """

    def decorator(obj: T) -> T:
        get_registry(enum_val).register(name)(obj)
        return obj

    return decorator


def build(kind: Any, name: str, *args, **kwargs):
    """Instantiate a registered object (or dotted import) by name.

    Args:
        kind (Any): Registry bucket (string or enum) to search.
        name (str): Name of the registered component or a fully‑qualified
            import path (e.g. ``"package.module.Class"``).
        *args: Positional arguments forwarded to the target constructor.
        **kwargs: Keyword arguments forwarded to the target constructor.

    Returns:
        Any: The instantiated object.

    Raises:
        KeyError: If ``name`` is not found in the registry and is not a valid
            import path.

    Example:
        >>> from lightning_ml.core.utils.registry import build, register
        >>> @register("model")
        ... class Linear:
        ...     def __init__(self, in_features: int, out_features: int):  # doctest: +SKIP
        ...         self.shape = (in_features, out_features)              # doctest: +SKIP
        >>> obj = build("model", "Linear", 3, 2)
        >>> obj.shape
        (3, 2)
    """
    if name is None:
        raise KeyError("cfg must have a `name` key")
    if not any(r.value == kind for r in Registries):
        raise KeyError(f"Unrecognized kind: {kind}")
    key = _to_key(kind)
    if "." in name:
        cls_or_fn = import_from_str(name)
    else:
        cls_or_fn = REGISTRIES[key][name]
    return cls_or_fn(*args, **kwargs)


def build_from_cfg(kind: Any, name: str, *args, **kwargs):
    """Instantiate a registered object using a configuration mapping.

    This convenience wrapper expects ``params`` to be provided in ``**kwargs``.
    If present, the mapping stored under ``params`` is unpacked and forwarded
    as keyword arguments to :func:`build`.

    Args:
        kind (Any): Registry bucket (string or enum) to search.
        name (str): Name of the registered component or a fully‑qualified
            import path.
        *args: Positional arguments forwarded to the target constructor.
        **kwargs: Should include an optional ``params`` mapping containing
            keyword arguments for the constructor. Any other keys are ignored.

    Returns:
        Any: The instantiated object.

    Example:
        >>> cfg = {"params": {"in_features": 3, "out_features": 2}}
        >>> obj = build_from_cfg("model", "Linear", **cfg)
        >>> obj.shape
        (3, 2)
    """
    params = kwargs.get("params", {})
    return build(kind, name, *args, **params)


@requires("hydra.utils", "omegaconf")
def instantiate_from_yaml(cfg_path: str | Path) -> Any:
    """Instantiate an object from a YAML config using Hydra.

    The YAML file must contain a ``_target_`` key. When using the
    :func:`build_from_cfg` helper this typically looks like::

        _target_: lightning_ml.core.utils.registry.build_from_cfg
        kind: dataset
        name: LabelledDataset
        params:
            inputs: [1, 2]
            targets: [3, 4]

    Parameters
    ----------
    cfg_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    Any
        The instantiated object as returned by ``hydra.utils.instantiate``.
    """

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(cfg_path))
    return instantiate(cfg)
