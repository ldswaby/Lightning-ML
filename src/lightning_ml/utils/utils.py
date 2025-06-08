from typing import Optional, Type, TypeVar, overload, Any, Union

BaseT = TypeVar("BaseT", bound=type)
MixinT = TypeVar("MixinT", bound=type)


__all__ = ["bind_classes"]


@overload
def bind_classes(
    base_cls: Type[BaseT],
    mixin_cls: None = None,
    name: str | None = ...,
) -> Type[BaseT]: ...


@overload
def bind_classes(
    base_cls: Type[BaseT],
    mixin_cls: Type[MixinT],
    name: str | None = ...,
) -> Type[Union[BaseT, MixinT]]: ...


def bind_classes(
    base_cls: Type[BaseT],
    mixin_cls: Optional[Type[MixinT]] = None,
    name: str | None = None,
) -> Type[Any]:
    """Return a new class that combines ``mixin_cls`` and ``base_cls``.

    The resulting class’ MRO is::

        (mixin_cls, base_cls, *base_cls.__mro__[1:])

    Args:
        base_cls: The “main” class that provides most behaviour/state.
        mixin_cls: A mix-in whose methods should override those of
            ``base_cls``.  Its implementation can still call
            ``super()`` to reach the base versions.
        name: Optional name for the generated class.  If ``None``,
            defaults to ``f"{base_cls.__name__}{mixin_cls.__name__}"``.

    Returns:
        Type[Any]: A new Python class object.

    Notes:
        * Works for any pair of classes, not just LightningModules.
        * Order matters: ``mixin_cls`` is placed **first** so its
          attributes shadow those of ``base_cls`` while remaining able
          to delegate via ``super()``.
    """
    if mixin_cls is None:
        return base_cls
    if name is None:
        name = f"{base_cls.__name__}{mixin_cls.__name__}"
    return type(name, (mixin_cls, base_cls), {})
