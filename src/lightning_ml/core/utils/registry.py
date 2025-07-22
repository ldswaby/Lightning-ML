"""Generic registry for components."""

from __future__ import annotations

from typing import Any, Callable, Dict, TypeVar

from .imports import import_from_str

__all__ = ["Registry", "REGISTRIES", "get_registry", "register", "build"]

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


def build(kind: Any, cfg: Dict[str, Any], *args, **kw):
    """Instantiate a registered object or dotted import based on a config.

    Args:
        kind: Registry bucket (string or enum) to search.
        cfg: Mapping with keys ``"name"`` and optional ``"params"``.
        *args: Positional arguments forwarded to the constructor.
        **kw: Keyword arguments forwarded to the constructor.

    Returns:
        Any: The instantiated object.

    Examples:
        >>> from lightning_ml.core.utils.registry import build, register
        >>> @register("model")
        ... class Linear:
        ...     def __init__(self, in_features: int, out_features: int):
        ...         self.shape = (in_features, out_features)
        >>> cfg = {"name": "Linear", "params": {"in_features": 3, "out_features": 2}}
        >>> obj = build("model", cfg)
        >>> obj.shape
        (3, 2)
    """
    name = cfg["name"]
    key = _to_key(kind)
    if "." in name:
        cls_or_fn = import_from_str(name)
    else:
        cls_or_fn = REGISTRIES[key][name]
    return cls_or_fn(*args, **cfg.get("params", {}), **kw)
