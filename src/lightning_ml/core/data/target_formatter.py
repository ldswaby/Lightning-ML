from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, List, Optional


@dataclass
class TargetFormatter(ABC):
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
    labels: Optional[List[str]] = None
    num_classes: Optional[int] = None

    def __post_init__(self):
        self.num_classes = (
            len(self.labels) if self.labels is not None else self.num_classes
        )

    def __call__(self, target: Any) -> Any:
        return self.format(target)

    @abstractmethod
    def format(self, target: Any) -> Any:
        """Format the target

        Args:
            target (Any): _description_

        Returns:
            Any: _description_
        """
