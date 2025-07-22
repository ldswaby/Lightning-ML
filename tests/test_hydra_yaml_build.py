import types
import sys
import importlib
from pathlib import Path
import yaml


def test_instantiate_from_yaml():
    # Stub minimal dependencies as in other tests
    torch = types.ModuleType('torch')
    torch.Tensor = object
    torch.nn = types.ModuleType('nn')
    torch.nn.Module = object
    torch.nn.ModuleDict = object
    torch.utils = types.ModuleType('utils'); torch.utils.data = types.ModuleType('data')
    torch.utils.data.Dataset = object
    torch.utils.data.DataLoader = object
    torch.utils.data.Subset = object
    torch.optim = types.ModuleType('optim')
    torch.optim.lr_scheduler = types.ModuleType('lr_scheduler')
    torch.optim.lr_scheduler._LRScheduler = object
    torch.optim.optimizer = types.ModuleType('optimizer')
    torch.optim.optimizer.Optimizer = object
    sys.modules['torch.optim'] = torch.optim
    sys.modules['torch.optim.lr_scheduler'] = torch.optim.lr_scheduler
    sys.modules['torch.optim.optimizer'] = torch.optim.optimizer
    sys.modules['torch'] = torch
    sys.modules['torch.nn'] = torch.nn
    sys.modules['torch.utils'] = torch.utils
    sys.modules['torch.utils.data'] = torch.utils.data

    pl = types.ModuleType('pytorch_lightning')
    pl.LightningModule = type('LightningModule', (), {})
    pl.LightningDataModule = type('LightningDataModule', (), {})
    pl.Trainer = type('Trainer', (), {})
    sys.modules['pytorch_lightning'] = pl

    sys.modules['pandas'] = types.ModuleType('pandas')
    sys.modules['numpy'] = types.ModuleType('numpy')
    PIL = types.ModuleType('PIL'); PIL_Image = types.ModuleType('PIL.Image')
    sys.modules['PIL'] = PIL; sys.modules['PIL.Image'] = PIL_Image
    sklearn = types.ModuleType('sklearn')
    sklearn_ms = types.ModuleType('sklearn.model_selection')
    sklearn_split = types.ModuleType('sklearn.model_selection._split')
    sklearn_split.BaseCrossValidator = object
    sys.modules['sklearn'] = sklearn
    sys.modules['sklearn.model_selection'] = sklearn_ms
    sys.modules['sklearn.model_selection._split'] = sklearn_split
    torchmetrics = types.ModuleType('torchmetrics')
    torchmetrics.MetricCollection = object
    sys.modules['torchmetrics'] = torchmetrics

    # Stub lightning_utilities
    lu = types.ModuleType('lightning_utilities')
    lu_core = types.ModuleType('core')
    lu_imports = types.ModuleType('imports')
    lu_imports.compare_version = lambda *a, **k: False
    lu_imports.module_available = lambda name: False
    lu_core.imports = lu_imports
    lu.core = lu_core
    sys.modules['lightning_utilities'] = lu
    sys.modules['lightning_utilities.core'] = lu_core
    sys.modules['lightning_utilities.core.imports'] = lu_imports

    # Stub hydra and omegaconf so hydra isn't required
    hydra = types.ModuleType('hydra')
    hydra_utils = types.ModuleType('hydra.utils')
    sys.path.insert(0, 'src')
    from lightning_ml.core.utils.imports import import_from_str

    def instantiate(cfg: dict):
        fn = import_from_str(cfg.pop('_target_'))
        return fn(**cfg)

    hydra_utils.instantiate = instantiate
    hydra.utils = hydra_utils
    sys.modules['hydra'] = hydra
    sys.modules['hydra.utils'] = hydra_utils

    oc = types.ModuleType('omegaconf')

    class _OC:
        @staticmethod
        def load(p):
            with open(p) as fh:
                return yaml.safe_load(fh)

    oc.OmegaConf = _OC
    sys.modules['omegaconf'] = oc

    datasets = importlib.import_module('lightning_ml.datasets')
    from lightning_ml.core.utils.registry import instantiate_from_yaml

    cfg_path = Path('config/dataset/labelled.yaml')

    ds = instantiate_from_yaml(cfg_path)
    assert isinstance(ds, datasets.LabelledDataset)
    assert len(ds) == 2


