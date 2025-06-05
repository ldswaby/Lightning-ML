"""Generic registry class for registering and retrieving components by name."""

from typing import Callable, List, Optional, TypeVar

T = TypeVar("T")


class Registry(dict):
    """Registry for storing classes under string keys.

    Attributes:
        _lib (str): Identifier for the type of components being registered (e.g., 'Model', 'Metric', 'Loss').
    """

    def __init__(self, lib: str):
        """Initializes the registry.

        Args:
            lib (str): A label for the registry to indicate what it stores.
        """
        super().__init__()
        self._lib = lib

    def register(self, name: Optional[str] = None) -> Callable[[T], T]:
        """Decorator for registering a class or function.

        Args:
            name (Optional[str]): Optional name to register under. If not provided, uses the class's name.

        Raises:
            KeyError: If the name is already registered.

        Returns:
            Callable[[T], T]: A decorator that registers the class or function.
        """

        def decorator(cls: T) -> T:
            key = name or cls.__name__
            if key in self:
                raise KeyError(
                    f"{self._lib.capitalize()} '{key}' is already registered."
                )
            self[key] = cls
            return cls

        return decorator

    def get(self, name: str) -> T:
        """Retrieves a registered class or function by name.

        Args:
            name (str): Name of the registered component.

        Raises:
            KeyError: If the name is not found in the registry.

        Returns:
            T: The registered class or function.
        """
        if name not in self:
            raise KeyError(
                f"{self._lib.capitalize()} '{name}' not found in the registry."
            )
        return super().__getitem__(name)

    def list_keys(self) -> List[str]:
        """Returns a list of all registered keys.

        Returns:
            List[str]: List of registered names.
        """
        return list(self.keys())
