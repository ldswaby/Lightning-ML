from typing import Any, Dict, Sequence

from lightning_ml.utils.data.splitters import DATA_SPLITTER_REGISTRY
from .mixins import HoldoutMixin, RandomHoldoutMixin


@DATA_SPLITTER_REGISTRY.register('random')
@DATA_SPLITTER_REGISTRY.register('holdout')
class RandomSplitter(RandomHoldoutMixin, HoldoutMixin):
    """Random holdout splitter."""

    # All logic provided by mixins
    pass
