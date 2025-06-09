from ..core import BaseDataset


class InputMixin(BaseDataset):
    """Dataset mix-in that mandates a `get_input` method."""

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
    pass


class LabelledDatasetBase(InputMixin, TargetMixin):
    pass


class ContastiveDatasetBase(InputMixin, PositiveMixin):
    pass


class TripletDatasetBase(ContastiveDatasetBase, NegativeMixin):
    pass
