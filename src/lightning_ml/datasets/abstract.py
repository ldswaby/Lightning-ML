"""
Abstract mix-in and base classes for dataset definitions.

This module provides:
- InputMixin, TargetMixin, PositiveMixin, NegativeMixin: enforce specific `get_` methods and sample_keys.
- UnlabelledDatasetBase, LabelledDatasetBase, ContastiveDatasetBase, TripletDatasetBase: abstract dataset base classes combining mix-ins.
"""

from ..core import BaseDataset

__all__ = [
    "InputMixin",
    "TargetMixin",
    "PositiveMixin",
    "NegativeMixin",
    "UnlabelledDatasetBase",
    "LabelledDatasetBase",
    "ContastiveDatasetBase",
    "TripletDatasetBase",
]


class InputMixin(BaseDataset):
    """Dataset mix-in that mandates a `get_input` method, the output of which
    will be added to the `__getitem__` ouput dict of all subclasses with key
    "input".
    """

    sample_keys = ["input"]


class TargetMixin(BaseDataset):
    """Dataset mix-in that mandates a `get_target` method."""

    sample_keys = ["target"]


class PositiveMixin(BaseDataset):
    """Dataset mix-in that mandates a `get_positive` method."""

    sample_keys = ["positive"]


class NegativeMixin(BaseDataset):
    """Dataset mix-in that mandates a `get_negative` method."""

    sample_keys = ["negative"]


# Abstract dataset bases


class UnlabelledDatasetBase(InputMixin):
    """
    Abstract base for unlabeled datasets.

    Combines mix-ins to enforce retrieval of input samples only.
    """


class LabelledDatasetBase(InputMixin, TargetMixin):
    """
    Abstract base for labeled datasets.

    Combines mix-ins to enforce retrieval of input and target samples.
    """


class ContastiveDatasetBase(InputMixin, PositiveMixin):
    """
    Abstract base for contrastive datasets.

    Combines mix-ins to enforce retrieval of input and positive samples.
    """


class TripletDatasetBase(ContastiveDatasetBase, NegativeMixin):
    """
    Abstract base for triplet datasets.

    Combines mix-ins to enforce retrieval of input, positive, and negative samples.
    """
