from lightning_ml.utils.data.splitters import DATA_SPLITTER_REGISTRY
from .mixins import KFoldMixin, StratifiedKFoldMixin


@DATA_SPLITTER_REGISTRY.register('stratified_kfold')
class StratifiedKFoldSplitter(StratifiedKFoldMixin, KFoldMixin):
    """Stratified K-Fold splitter."""

    pass
