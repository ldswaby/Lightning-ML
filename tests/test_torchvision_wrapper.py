import sys
import types
import importlib


def test_torchvision_dataset_wrapper():
    # Stub torch and dependencies
    torch = types.ModuleType('torch')
    torch.Tensor = object
    torch.nn = types.ModuleType('nn')
    torch.nn.Module = object
    torch.nn.ModuleDict = object
    torch.utils = types.ModuleType('utils'); torch.utils.data = types.ModuleType('data')
    class BaseDataset(object):
        pass
    torch.utils.data.Dataset = BaseDataset
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

    # Define dummy torchvision dataset
    tv = types.ModuleType('torchvision')
    tv_datasets = types.ModuleType('torchvision.datasets')

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.data = [(1, 'a'), (2, 'b')]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    tv_datasets.DummyDataset = DummyDataset
    tv.datasets = tv_datasets
    sys.modules['torchvision'] = tv
    sys.modules['torchvision.datasets'] = tv_datasets

    sys.path.insert(0, 'src')
    datasets = importlib.import_module('lightning_ml.datasets')

    ds = datasets.TorchvisionDataset(DummyDataset())
    assert len(ds) == 2
    assert ds.get_input(0) == 1
    assert ds.get_target(1) == 'b'

