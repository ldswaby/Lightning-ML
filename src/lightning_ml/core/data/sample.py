from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from ..utils.enums import DataKeys


@dataclass(frozen=True, slots=True)
class Sample:  # noqa: D101
    """Container for a single example travelling through the ML pipeline.

    This lightweight dataclass groups together the canonical pieces of data that a
    training or inference step may need. At minimum, each ``Sample`` **must** carry
    the raw model input. All other attributes are optional and can be populated
    lazily (for example, attaching ``preds`` after a forward pass).

    Attributes:
        input: Raw input that will be fed to the model (for example, an image
            tensor or token IDs).
        preds: Model predictions - typically filled in by the evaluation loop.
        target: Ground-truth label(s) used for supervised learning.
        positive: Positive sample (for example, in contrastive learning).
        negative: Negative sample (for example, in contrastive learning).
        metadata: Arbitrary auxiliary information such as file paths or sample IDs.
    """

    input: Any
    preds: Optional[Any] = None
    target: Optional[Any] = None
    positive: Optional[Any] = None
    negative: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __getitem__(self, key: str | DataKeys) -> Any:
        if isinstance(key, str):
            pass
        elif isinstance(key, DataKeys):
            key = key.value
        else:
            raise KeyError(f"Key must be str or DataKeys enum, got {type(key)}.")
        return getattr(self, key)

    def keys(self, include_none: bool = False) -> Sequence[DataKeys]:
        """Return the keys present in the sample.

        Args:
            include_none: If ``True``, list **all** keys irrespective of their
                value; otherwise, include only those whose value is not ``None``.

        Returns:
            Sequence[DataKeys]: A sequence of keys present in the sample.
        """
        if include_none:
            return list(DataKeys)
        return [k for k in DataKeys if getattr(self, k.value) is not None]

    def to_dict(self, include_none: bool = False) -> Dict[str, Any]:
        """Return a plain ``dict`` representation of the sample.

        Args:
            include_none: If ``True``, include keys whose value is ``None``.

        Returns:
            Dict[str, Any]: A dictionary keyed by the names of :class:`DataKeys`.
        """
        data = {k.value: getattr(self, k.value) for k in DataKeys}
        if not include_none:
            data = {k: v for k, v in data.items() if v is not None}
        return data
