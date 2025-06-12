from lightning_ml.utils.data.splitters import DATA_SPLITTER_REGISTRY
from .mixins import KFoldMixin, RandomKFoldMixin


@DATA_SPLITTER_REGISTRY.register('kfold')
class KFoldSplitter(RandomKFoldMixin, KFoldMixin):
    """Standard K-Fold splitter."""

    pass
