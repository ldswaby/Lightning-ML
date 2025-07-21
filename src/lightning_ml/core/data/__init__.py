"""Convenience re-exports for data abstractions."""

__all__ = ["BaseDataset", "BaseLoader", "Sample"]


def __getattr__(name: str):
    if name == "BaseDataset":
        from ..abstract.dataset import BaseDataset
        return BaseDataset
    if name == "BaseLoader":
        from ..abstract.loader import BaseLoader
        return BaseLoader
    if name == "Sample":
        from ..abstract.sample import Sample
        return Sample
    raise AttributeError(name)
