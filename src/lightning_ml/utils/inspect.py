"""
Module: inspect_utils
Description: A collection of utilities for Python runtime inspection and reflection.
"""

import importlib
import inspect
from dataclasses import fields, is_dataclass
from types import ModuleType
from typing import Any, Callable, List, Type, Union

__all__ = [
    "get_parent_classes",
    "get_child_classes",
    "get_module_functions",
    "import_by_path",
    "get_signature",
    "get_summary",
    "ensure_has_attributes",
    "get_dataclass_fields",
    "get_class_parameters",
]


# Private helper to DRY module loading
def _load_module(module: Union[str, ModuleType]) -> ModuleType:
    """
    Load and return a module object from a module name or return if already a module.
    """
    if isinstance(module, str):
        return importlib.import_module(module)
    if isinstance(module, ModuleType):
        return module
    raise TypeError("module must be a module object or importable module name")


def get_parent_classes(
    target: Union[Type[Any], Any],
    module: Union[str, ModuleType],
    recursive: bool = True,
) -> List[Type[Any]]:
    """
    Return all base classes of `target` (a class or instance) that are defined in `module`.

    Args:
        target: the class or instance whose base classes to find.
        module: module object or importable module name (e.g. "sklearn.model_selection").
        recursive: if True, include the full MRO (minus the class itself);
                   if False, only direct bases.

    Returns:
        A list of base-class types from the specified module.
    """
    # Normalize to a class
    cls = target if inspect.isclass(target) else target.__class__

    # Load module
    module_obj = _load_module(module)

    # Choose which bases to inspect
    bases = cls.__mro__[1:] if recursive else cls.__bases__

    # Filter bases by module
    return [
        base
        for base in bases
        if getattr(base, "__module__", None) == module_obj.__name__
    ]


def get_child_classes(
    base: Type[Any], module: Union[str, ModuleType]
) -> List[Type[Any]]:
    """
    Return all subclasses of `base` defined in `module`, excluding the base itself.

    Args:
        base: the base class to inspect.
        module: module object or importable module name.

    Returns:
        A list of subclass types.
    """
    module_obj = _load_module(module)

    subclasses = []
    for _, obj in inspect.getmembers(module_obj, inspect.isclass):
        if issubclass(obj, base) and obj is not base:
            subclasses.append(obj)
    return subclasses


def get_module_functions(module: Union[str, ModuleType]) -> List[Callable]:
    """
    Return all top-level functions defined in `module`.

    Args:
        module: module object or importable module name.

    Returns:
        A list of function objects.
    """
    module_obj = _load_module(module)

    return [
        fn
        for _, fn in inspect.getmembers(module_obj, inspect.isfunction)
        if fn.__module__ == module_obj.__name__
    ]


def import_by_path(path: str) -> Any:
    """
    Dynamically import a module or attribute by string path.

    Args:
        path: either 'package.module' or 'package.module:attribute'.

    Returns:
        The imported module or attribute.

    Raises:
        ImportError if the module or attribute cannot be imported.
    """
    try:
        if ":" in path:
            module_name, attr = path.split(":", 1)
            module_obj = importlib.import_module(module_name)
            return getattr(module_obj, attr)
        return importlib.import_module(path)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Cannot import '{path}': {e}") from e


def get_signature(obj: Any) -> str:
    """Return the signature of a function or class as a string."""
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return ""


def get_summary(obj: Any) -> str:
    """Get the first line of the docstring for `obj`, or an empty string."""
    doc = inspect.getdoc(obj) or ""
    return doc.strip().split("\n", 1)[0]


def ensure_has_attributes(obj: Any, attrs: List[str]) -> None:
    """
    Raise AttributeError if `obj` is missing any of the specified attribute names.
    """
    missing = [a for a in attrs if not hasattr(obj, a)]
    if missing:
        raise AttributeError(f"{obj!r} is missing attributes: {missing}")


def get_dataclass_fields(obj: Any) -> List[str]:
    """
    If `obj` is a dataclass or instance thereof, return its field names.

    Raises:
        TypeError if `obj` is not a dataclass.
    """
    cls = obj if inspect.isclass(obj) else obj.__class__
    if is_dataclass(cls):
        return [f.name for f in fields(cls)]
    raise TypeError(f"{cls.__name__} is not a dataclass")


def get_class_parameters(obj: Any) -> List[str]:
    """
    If `obj` is a class or instance thereof, return its parameter names.
    """
    cls = obj if inspect.isclass(obj) else obj.__class__
    sig = inspect.signature(cls)
    return list(sig.parameters)
