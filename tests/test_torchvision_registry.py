import sys
import types


def test_register_torchvision_modules():
    # Stub torch and dependencies as in existing tests
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

    # Stub torchvision with simple dataset and model
    tv = types.ModuleType('torchvision')
    tv_datasets = types.ModuleType('torchvision.datasets')
    tv_models = types.ModuleType('torchvision.models')

    class DummyDataset(torch.utils.data.Dataset):
        pass

    class DummyModel(torch.nn.Module):
        pass

    def dummy_model_fn():
        return 'model'

    tv_datasets.DummyDataset = DummyDataset
    tv_models.DummyModel = DummyModel
    tv_models.dummy_model_fn = dummy_model_fn

    tv.datasets = tv_datasets
    tv.models = tv_models
    sys.modules['torchvision'] = tv
    sys.modules['torchvision.datasets'] = tv_datasets
    sys.modules['torchvision.models'] = tv_models

    sys.path.insert(0, 'src')

    from lightning_ml.datasets import DATASET_REG
    from lightning_ml.models import MODEL_REG
    from lightning_ml.utils.torchvision import register_torchvision

    DATASET_REG.clear()
    MODEL_REG.clear()

    register_torchvision()

    assert 'DummyDataset' in DATASET_REG
    assert 'DummyModel' in MODEL_REG
    assert 'dummy_model_fn' in MODEL_REG
