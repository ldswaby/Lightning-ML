"""Generic registry for components."""

from __future__ import annotations

from typing import Any, Callable, Dict, TypeVar

from .imports import import_from_str

__all__ = ["Registry", "get_registry", "register"]

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

        Raises:
            KeyError: If ``name`` is not found in the registry.
        """
        if name not in self:
            raise KeyError(f"{self._lib} '{name}' not found in registry")
        return super().__getitem__(name)

    def list_keys(self) -> list[str]:  # pragma: no cover - trivial
        """Return the list of registered component names."""
        return list(self.keys())


# Global enum-based registries

# Internal mapping of enum values to Registry instances
_REGISTRIES: dict[Any, Registry] = {}


def get_registry(enum_val: Any) -> Registry:
    """Get or create the :class:`Registry` associated with an enum value.

    Args:
        enum_val: The enum value that identifies the registry.

    Returns:
        Registry: The corresponding registry instance.
    """
    return _REGISTRIES.setdefault(enum_val, Registry(enum_val.value))


def register(enum_val: Any, name: str | None = None) -> Callable[[T], T]:
    """Return a decorator that registers a class or callable to the registry indicated by ``enum_val``.

    @register(Registries.MODEL)
    ...

    Args:
        enum_val: Enum value that identifies the registry.
        name: Optional explicit name to register the object under.

    Returns:
        Callable[[T], T]: A decorator that registers the object.
    """

    def decorator(obj: T) -> T:
        get_registry(enum_val).register(name)(obj)
        return obj

    return decorator


def build(kind: str, cfg: Dict[str, Any], *args, **kw):
    """Generic factory: cfg = {'name': 'mlp', 'params': {...}}"""
    name = cfg["name"]
    if "." in name:
        cls_or_fn = import_from_str(name)
    else:
        cls_or_fn = _REGISTRIES[kind][name]
    return cls_or_fn(*args, **cfg.get("params", {}), **kw)
