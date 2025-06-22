import sys
import types
import importlib


def test_numpy_labelled_dataset_instantiation():
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
    datasets = importlib.import_module('lightning_ml.datasets')

    ds = datasets.NumpyLabelledDataset([1, 2], [3, 4])
    assert len(ds) == 2

