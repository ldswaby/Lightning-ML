import sys
import types
import importlib.util
import pathlib

BASE = pathlib.Path('src/lightning_ml')

# Helper to load module by path without executing package __init__
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_numpy_labelled_dataset_instantiation():
    # Stub minimal dependencies
    torch = types.ModuleType('torch')
    torch.Tensor = object
    torch.nn = types.ModuleType('nn'); torch.nn.Module = object
    torch.utils = types.ModuleType('utils'); torch.utils.data = types.ModuleType('data'); torch.utils.data.Dataset = object
    sys.modules['torch'] = torch
    sys.modules['torch.nn'] = torch.nn
    sys.modules['torch.utils'] = torch.utils
    sys.modules['torch.utils.data'] = torch.utils.data

    pl = types.ModuleType('pytorch_lightning')
    pl.LightningModule = type('LightningModule', (), {})
    sys.modules['pytorch_lightning'] = pl

    sys.modules['pandas'] = types.ModuleType('pandas')
    sys.modules['numpy'] = types.ModuleType('numpy')
    PIL = types.ModuleType('PIL'); PIL_Image = types.ModuleType('PIL.Image')
    sys.modules['PIL'] = PIL; sys.modules['PIL.Image'] = PIL_Image

    # Load modules manually
    core_dataset = _load('lightning_ml.core.dataset', BASE / 'core/dataset.py')
    core_pkg = types.ModuleType('lightning_ml.core')
    core_pkg.BaseDataset = core_dataset.BaseDataset
    sys.modules['lightning_ml.core'] = core_pkg

    _load('lightning_ml.datasets.abstract', BASE / 'datasets/abstract.py')
    _load('lightning_ml.datasets.unlabelled', BASE / 'datasets/unlabelled.py')
    _load('lightning_ml.datasets.labelled', BASE / 'datasets/labelled.py')
    disk = _load('lightning_ml.datasets.disk', BASE / 'datasets/disk.py')

    ds = disk.NumpyLabelledDataset([1, 2], [3, 4])
    assert len(ds) == 2

