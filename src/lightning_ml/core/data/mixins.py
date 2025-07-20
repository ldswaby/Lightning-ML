from ..abstract.dataset import BaseDataset
from ..utils.enums import DataKeys


class InputMixin(BaseDataset):
    """Dataset mix-in that mandates a `get_input` method, the output of which
    will be added to the `__getitem__` ouput dict of all subclasses with key
    "input".
    """

    sample_keys = [DataKeys.INPUT]


class TargetMixin(BaseDataset):
    """Dataset mix-in that mandates a `get_target` method."""

    sample_keys = [DataKeys.TARGET]


class PositiveMixin(BaseDataset):
    """Dataset mix-in that mandates a `get_positive` method."""

    sample_keys = [DataKeys.POSITIVE]


class NegativeMixin(BaseDataset):
    """Dataset mix-in that mandates a `get_negative` method."""

    sample_keys = [DataKeys.NEGATIVE]
