import re
from dataclasses import dataclass
from functools import reduce
from typing import Any, ClassVar, Optional, Type, cast

import numpy as np
import torch
from torch import Tensor

from ..utils.sort import sorted_alphanumeric


def _is_list_like(x: Any) -> bool:
    try:
        _ = x[0]
        _ = len(x)
        return True
    except (TypeError, IndexError, KeyError):
        return False


def _as_list(x: list | Tensor | np.ndarray) -> list[Any]:
    if torch.is_tensor(x) or isinstance(x, np.ndarray):
        return cast(list[Any], x.tolist())
    return x


def _strip(x: str | int) -> str:
    """Replace both ` ` and `,` from str."""
    if isinstance(x, str):
        return x.strip(", ")
    return str(x)


@dataclass
class TargetFormatter:
    """A ``TargetFormatter`` is used to convert targets of a given type to a standard format required by the loss
    function. To implement a custom ``TargetFormatter``, simply override the ``format`` method with your own logic.

    Examples
    ________

    .. doctest::

        >>> from dataclasses import dataclass
        >>> from typing import ClassVar, Optional
        >>> from flash.core.data.utilities.classification import TargetFormatter
        >>>
        >>> @dataclass
        ... class CustomStringTargetFormatter(TargetFormatter):
        ...     "A ``TargetFormatter`` which converts strings of the format '#<index>' to integers."
        ...     multi_label: ClassVar[Optional[bool]] = False
        ...     def format(self, target: str) -> int:
        ...         return int(target.strip("#"))
        ...
        >>> formatter = CustomStringTargetFormatter()
        >>> formatter("#1")
        1

    """

    multi_label: ClassVar[Optional[bool]] = None
    numeric: ClassVar[Optional[bool]] = None
    binary: ClassVar[Optional[bool]] = None
    labels: list[str] | None = None
    num_classes: int | None = None

    def __post_init__(self):
        self.num_classes = (
            len(self.labels) if self.labels is not None else self.num_classes
        )

    def __call__(self, target: Any) -> Any:
        return self.format(target)

    def format(self, target: Any) -> Any:
        raise NotImplementedError


@dataclass
class SingleNumeric(TargetFormatter):
    """A ``SingleNumeric`` for targets that contain a single numeric value (the class index).

    Examples
    ________

    .. doctest::

        >>> import torch
        >>> from flash.core.data.utilities.classification import SingleNumeric
        >>> formatter = SingleNumeric(num_classes=10)
        >>> formatter(5)
        5
        >>> formatter([5])
        5
        >>> formatter(torch.tensor(5))
        5

    """

    multi_label: ClassVar[Optional[bool]] = False
    numeric: ClassVar[Optional[bool]] = True
    binary: ClassVar[Optional[bool]] = False

    def format(self, target: Any) -> Any:
        result = _as_list(target)
        if _is_list_like(result):
            result = result[0]
        return result


@dataclass
class SingleLabel(TargetFormatter):
    """A ``SingleLabel`` for targets that contain a single string label.

    Examples
    ________

    .. doctest::

        >>> from flash.core.data.utilities.classification import SingleLabel
        >>> formatter = SingleLabel(labels=["cat", "dog"], num_classes=2)
        >>> formatter("cat")
        0
        >>> formatter(["dog"])
        1

    """

    multi_label: ClassVar[Optional[bool]] = False
    numeric: ClassVar[Optional[bool]] = False
    binary: ClassVar[Optional[bool]] = False

    def __post_init__(self):
        super().__post_init__()
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

    def format(self, target: Any) -> Any:
        return self.label_to_idx[
            _strip(
                target[0]
                if _is_list_like(target) and not isinstance(target, str)
                else target
            )
        ]


@dataclass
class SingleBinary(TargetFormatter):
    """A ``SingleBinary`` for targets that are one-hot encoded binaries.

    Examples
    ________

    .. doctest::

        >>> import torch
        >>> from flash.core.data.utilities.classification import SingleBinary
        >>> formatter = SingleBinary(num_classes=2)
        >>> formatter([1, 0])
        0
        >>> formatter(torch.tensor([0, 1]))
        1

    """

    multi_label: ClassVar[Optional[bool]] = False
    numeric: ClassVar[Optional[bool]] = False
    binary: ClassVar[Optional[bool]] = True

    def format(self, target: Any) -> Any:
        for idx, t in enumerate(target):
            if t == 1:
                return idx
        return 0


@dataclass
class MultiNumeric(TargetFormatter):
    """A ``MultiNumeric`` for targets that contain multiple numeric values (the class indices).

    Examples
    ________

    .. doctest::

        >>> import torch
        >>> from flash.core.data.utilities.classification import MultiNumeric
        >>> formatter = MultiNumeric(num_classes=10)
        >>> formatter([2, 5])
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
        >>> formatter(torch.tensor([2, 5]))
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0]

    """

    multi_label: ClassVar[Optional[bool]] = True
    numeric: ClassVar[Optional[bool]] = True
    binary: ClassVar[Optional[bool]] = False

    def format(self, target: Any) -> Any:
        result = [0] * self.num_classes
        for idx in target:
            result[idx] = 1
        return result


@dataclass
class MultiLabel(SingleLabel):
    """A ``MultiLabel`` for targets that contain multiple string labels. Provide ``delimiter`` if the targets come as a single string; otherwise pass a list of labels.

    Examples:

        >>> from flash.core.data.utilities.classification import MultiLabel
        >>> formatter = MultiLabel(labels=["bird", "cat", "dog"], num_classes=3)
        >>> formatter(["cat", "dog"])
        [0, 1, 1]
        >>> formatter = MultiLabel(labels=["bird", "cat", "dog"], num_classes=3, delimiter=',')
        >>> formatter("cat,dog")
        [0, 1, 1]
    """

    multi_label: ClassVar[Optional[bool]] = True
    numeric: ClassVar[Optional[bool]] = False
    binary: ClassVar[Optional[bool]] = False
    delimiter: str | None = None

    def format(self, target: Any) -> Any:
        # If a single string is provided, convert to a list first
        if isinstance(target, str):
            if self.delimiter:
                target = target.split(self.delimiter)
            else:
                target = re.split(r"[,\s]+", target.strip())
        result = [0] * self.num_classes
        for t in target:
            idx = super().format(t)
            result[idx] = 1
        return result


@dataclass
class MultiBinary(TargetFormatter):
    """A ``MultiBinary`` for targets that are multi-hot binary.

    Examples
    ________

    .. doctest::

        >>> import torch
        >>> from flash.core.data.utilities.classification import MultiBinary
        >>> formatter = MultiBinary(num_classes=3)
        >>> formatter([0, 1, 1])
        [0, 1, 1]
        >>> formatter(torch.tensor([1, 0, 0]))
        [1, 0, 0]

    """

    multi_label: ClassVar[Optional[bool]] = True
    numeric: ClassVar[Optional[bool]] = False
    binary: ClassVar[Optional[bool]] = True

    def format(self, target: Any) -> Any:
        return _as_list(target)


@dataclass
class MultiSoft(MultiBinary):
    """A ``MultiSoft`` for mutli-label soft targets.

    Examples
    ________

    .. doctest::

        >>> import torch
        >>> from flash.core.data.utilities.classification import MultiSoft
        >>> formatter = MultiSoft(num_classes=3)
        >>> formatter([0.1, 0.9, 0.6])
        [0.1, 0.9, 0.6]
        >>> formatter(torch.tensor([0.9, 0.6, 0.7]))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [0..., 0..., 0...]

    """

    binary: ClassVar[Optional[bool]] = False


def _get_target_formatter_type(target: Any) -> Type[TargetFormatter]:
    """Determine the ``TargetFormatter`` type for a given target.

    Multi-label targets can be:
        * String with delimiters - ``MultiLabel`` (e.g. ["blue,green", "red"] or ["blue green", "red"])
        * List of strings - ``MultiLabel`` (e.g. [["blue", "green"], ["red"]])
        * List of numbers - ``MultiNumeric`` (e.g. [[0, 1], [2]])
        * Binary list - ``MultiBinary`` (e.g. [[1, 1, 0], [0, 0, 1]])
        * Soft target - ``MultiSoft`` (e.g. [[0.1, 0, 0], [0.9, 0.7, 0]])

    Single-label targets can be:
        * Single string - ``SingleLabel`` (e.g. ["blue", "green", "red"])
        * Single number - ``SingleNumeric`` (e.g. [0, 1, 2])
        * One-hot binary list - ``SingleBinary`` (e.g. [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    Args:
        target: A target that is one of: a single target, a list of targets, a delimited string.

    """
    if isinstance(target, str):
        target = _strip(target)
        if "," in target or " " in target:
            return MultiLabel
        return SingleLabel
    if _is_list_like(target):
        if isinstance(target[0], str):
            return MultiLabel
        target = _as_list(target)
        if len(target) > 1:
            if all(t == 0 or t == 1 for t in target):
                if sum(target) == 1:
                    return SingleBinary
                return MultiBinary
            if any(isinstance(t, float) for t in target):
                return MultiSoft
            return MultiNumeric
    return SingleNumeric


_RESOLUTION_MAPPING: dict[Type[TargetFormatter], list[Type[TargetFormatter]]] = {
    MultiBinary: [MultiNumeric, MultiSoft],
    SingleBinary: [
        MultiBinary,
        MultiNumeric,
        MultiSoft,
    ],
    SingleLabel: [
        MultiLabel,
    ],
    SingleNumeric: [
        SingleBinary,
        MultiNumeric,
    ],
}


def _resolve_target_formatter(
    a: Type[TargetFormatter], b: Type[TargetFormatter]
) -> Type[TargetFormatter]:
    """The purpose of this resolution function is to enable reduction of the ``TargetFormatter`` type over multiple
    targets. For example, if one target formatter type is ``CommaDelimitedMultiLabel`` and the other type
    is ``SingleLabel`` then their reduction will be ``CommaDelimitedMultiLabel``.

    Raises:
        ValueError: If the two target formatters could not be resolved.

    """
    if a is b:
        return a
    if a in _RESOLUTION_MAPPING and b in _RESOLUTION_MAPPING[a]:
        return b
    if b in _RESOLUTION_MAPPING and a in _RESOLUTION_MAPPING[b]:
        return a
    raise ValueError(
        "Found inconsistent target formats. All targets should be either: single values, lists of values, or "
        "comma-delimited strings."
    )


def _get_target_details(
    targets: list[Any],
    target_formatter_type: Type[TargetFormatter],
) -> tuple[list[Any] | None, int]:
    """Given a list of targets and their ``TargetFormatter`` type, this function determines the ``labels`` and
    ``num_classes``. Targets can be:

    * Token-based: ``labels`` is the unique tokens, ``num_classes`` is the number of unique tokens.
    * Numeric: ``labels`` is ``None`` and ``num_classes`` is the maximum value plus one.
    * Binary: ``labels`` is ``None`` and ``num_classes`` is the length of the binary target.

    Args:
        targets: A list of targets.
        target_formatter_type: The ``TargetFormatter`` type.

    Returns:
        (labels, num_classes): Tuple containing the inferred ``labels`` (or ``None`` if no labels could be inferred)
        and ``num_classes``.

    """
    targets = _as_list(targets)
    if target_formatter_type.numeric:
        # Take a max over all values
        if target_formatter_type is MultiNumeric:
            values = []
            for target in targets:
                values.extend(target)
        else:
            values = targets
        num_classes = _as_list(max(values))
        if _is_list_like(num_classes):
            num_classes = num_classes[0]
        num_classes += 1
        labels = None
    elif target_formatter_type.binary or (target_formatter_type is MultiSoft):
        # Take a length
        # TODO: Add a check here and error if target lengths are not all equal
        num_classes = len(targets[0])
        labels = None
    else:
        # Compute tokens
        tokens = []
        if target_formatter_type is MultiLabel:
            for target in targets:
                if isinstance(target, str):
                    tokens.extend(re.split(r"[,\s]+", target.strip()))
                else:
                    tokens.extend(target)
        else:
            tokens = targets

        tokens = [_strip(token) for token in tokens]
        labels = list(sorted_alphanumeric(set(tokens)))
        num_classes = None
    return labels, num_classes


def get_target_formatter(
    targets: list[Any],
    labels: list[str] | None = None,
    num_classes: int | None = None,
    add_background: bool = False,
) -> TargetFormatter:
    """Get the ``TargetFormatter`` object to use for the given targets.

    Args:
        targets: The list of targets to format.
        labels: Optionally provide ``labels`` / ``num_classes`` instead of inferring them.
        num_classes: Optionally provide ``labels`` / ``num_classes`` instead of inferring them.
        add_background: If ``True``, a background class will be inserted as class zero if ``labels`` and
                ``num_classes`` are being inferred.

    Returns:
        The target formatter to use when formatting targets.

    """
    targets = _as_list(targets)
    target_formatter_type: Type[TargetFormatter] = reduce(
        _resolve_target_formatter,
        [_get_target_formatter_type(target) for target in targets],
    )
    if labels is None and num_classes is None:
        labels, num_classes = _get_target_details(targets, target_formatter_type)
        if add_background:
            labels = ["background"] + labels if labels is not None else labels
            num_classes = num_classes + 1 if num_classes is not None else num_classes
    return target_formatter_type(labels=labels, num_classes=num_classes)
