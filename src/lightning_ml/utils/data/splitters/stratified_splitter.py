from lightning_ml.utils.data.splitters import DATA_SPLITTER_REGISTRY
from .mixins import HoldoutMixin, StratifiedHoldoutMixin


@DATA_SPLITTER_REGISTRY.register('stratified')
class StratifiedSplitter(StratifiedHoldoutMixin, HoldoutMixin):
    """Stratified holdout splitter."""

    pass
