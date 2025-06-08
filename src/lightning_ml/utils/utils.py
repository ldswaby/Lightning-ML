from typing import Optional, Type, TypeVar, overload, Any, Union

BaseT = TypeVar("BaseT", bound=type)
MixinT = TypeVar("MixinT", bound=type)


__all__ = ["bind_classes", "bind"]


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


@overload
def bind(obj: BaseT, mixin: None = None, name: str | None = ...) -> BaseT:
    ...


@overload
def bind(obj: BaseT, mixin: MixinT, name: str | None = ...) -> Any:
    ...


def bind(obj: BaseT, mixin: Optional[MixinT] = None, name: str | None = None) -> Any:
    """Mutate ``obj`` so that its class incorporates ``mixin``.

    This is the instance-level analogue of :func:`bind_classes`.  The object's
    ``__class__`` is replaced with a dynamically created subclass that mixes in
    ``mixin`` so that its methods (e.g. ``predict_step``) override those of the
    original class.  Attributes stored on ``mixin`` are shallow-copied onto the
    target instance.

    Args:
        obj: The object to be modified in-place.
        mixin: The mix-in instance whose behaviour should override ``obj``.
        name: Optional name for the generated subclass.

    Returns:
        The modified ``obj``.
    """

    if mixin is None:
        return obj

    new_cls = bind_classes(obj.__class__, mixin.__class__, name)
    obj.__class__ = new_cls
    obj.__dict__.update(vars(mixin))
    return obj
