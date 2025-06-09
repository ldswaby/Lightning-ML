"""Utility helpers for the library."""

from __future__ import annotations

from typing import Type, TypeVar

BaseT = TypeVar("BaseT", bound=type)
MixinT = TypeVar("MixinT", bound=type)
OutT = TypeVar("OutT", bound=type)

__all__ = ["bind_classes"]


def bind_classes(base_cls: Type[BaseT], mixin_cls: Type[MixinT], name: str | None = None) -> Type[OutT]:
    """Create a new class combining ``mixin_cls`` and ``base_cls``.

    Parameters
    ----------
    base_cls : type
        Primary class providing most behaviour.
    mixin_cls : type
        Mixin whose methods override ``base_cls``.
    name : str, optional
        Name for the generated class. Defaults to ``"{base_cls.__name__}{mixin_cls.__name__}"``.

    Returns
    -------
    type
        Newly generated class with a method resolution order of
        ``(mixin_cls, base_cls, *base_cls.__mro__[1:])``.
    """
    if name is None:
        name = f"{base_cls.__name__}{mixin_cls.__name__}"
    return type(name, (mixin_cls, base_cls), {})

