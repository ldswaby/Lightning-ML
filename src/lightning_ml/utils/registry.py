"""Generic registry for components."""

from __future__ import annotations

from typing import List, Optional, TypeVar
from collections.abc import Callable

T = TypeVar("T")


class Registry(dict):
    """Dictionary-like registry of callables or classes."""

    def __init__(self, lib: str) -> None:
        """Create a new registry.

        Parameters
        ----------
        lib : str
            A human readable label describing what is being registered.
        """
        super().__init__()
        self._lib = lib

    # ------------------------------------------------------------------
    def register(self, name: str | None = None) -> Callable[[T], T]:
        """Decorator used to register ``cls`` under ``name``."""

        def decorator(cls: T) -> T:
            key = name or cls.__name__
            if key in self:
                raise KeyError(f"{self._lib} '{key}' is already registered")
            self[key] = cls
            return cls

        return decorator

    # ------------------------------------------------------------------
    def get(self, name: str) -> T:
        """Retrieve a registered component."""
        if name not in self:
            raise KeyError(f"{self._lib} '{name}' not found in registry")
        return super().__getitem__(name)

    def list_keys(self) -> list[str]:  # pragma: no cover - trivial
        """Return a list of registered names."""
        return list(self.keys())

