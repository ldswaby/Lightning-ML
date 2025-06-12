from ...registry import Registry

DATA_SPLITTER_REGISTRY = Registry('DataSplitter')

from .base_splitter import DataSplitter
from .holdout_splitter import RandomSplitter
from .stratified_splitter import StratifiedSplitter
from .kfold_splitter import KFoldSplitter
from .stratified_kfold_splitter import StratifiedKFoldSplitter


def _data_splitter_factory(config: dict) -> DataSplitter:
    """Instantiate a splitter from a config dictionary."""
    splitter = DATA_SPLITTER_REGISTRY.get(
        config['DATASET']['split_method']['name']
    )(
        **config['DATASET']['split_method']['kwargs']
    )
    return splitter
