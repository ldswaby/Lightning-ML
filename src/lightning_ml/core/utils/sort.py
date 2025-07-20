import re
from typing import Iterable, List, Union


def _convert(text: str) -> Union[int, str]:
    return int(text) if text.isdigit() else text


def _alphanumeric_key(key: str) -> List[Union[int, str]]:
    return [_convert(c) for c in re.split("([0-9]+)", key)]


def sorted_alphanumeric(iterable: Iterable[str]) -> Iterable[str]:
    """Sort the given iterable in the way that humans expect. For example, given ``{"class_1", "class_11", "class_2"}``
    this returns ``["class_1", "class_2", "class_11"]``.

    Copied from: https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/

    """
    return sorted(iterable, key=_alphanumeric_key)
