import sys
import types

from hydra.utils import instantiate


def test_hydra_registry_instantiation():
    # Stub minimal dependencies
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

    sys.path.insert(0, 'src')
    from lightning_ml.config import RegistryConfig
    from lightning_ml.core.utils.enums import Registries
    from lightning_ml.core.utils.registry import get_registry

    reg = get_registry(Registries.DATASET)
    reg.clear()

    class DummyDataset:
        def __init__(self, val):
            self.val = val

    reg.register('DummyDataset')(DummyDataset)

    cfg = RegistryConfig(registry=Registries.DATASET, name='DummyDataset', params={'val': 123})
    obj = instantiate(cfg)

    assert isinstance(obj, DummyDataset)
    assert obj.val == 123

