"""
Module for abstract dataset definitions with dynamic abstract methods.

This module defines:
- AutoAbstractMeta: metaclass that auto-generates abstract stubs based on sample_keys.
- BaseDataset: a generic PyTorch Dataset using dynamic getters.
- MyContrastiveDataset: example subclass for contrastive learning paradigms.
"""

from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset

__all__ = ["BaseDataset"]


class DatasetMeta(ABCMeta):
    """
    Metaclass that auto-injects abstract getter stubs for each key in `sample_keys`.

    If a subclass defines `sample_keys`, any missing `get_<key>` methods
    will be created as abstract stubs that raise NotImplementedError.

    NOTE: This enables further abtsract Dataset classes to be created simply from list of required sample keys
    """

    def __new__(
        mcls, name: str, bases: Tuple[type, ...], namespace: Dict[str, Any]
    ) -> "DatasetMeta":
        """
        Construct a new class, injecting abstractmethod stubs.

        Args:
            name (str): Name of the new class.
            bases (Tuple[type, ...]): Base classes of the new class.
            namespace (Dict[str, Any]): Attributes defined in the class body.

        Returns:
            AutoAbstractMeta: The newly created class with injected stubs.
        """
        # Determine declared and inherited sample_keys, then form their union
        declared_keys: List[str] = namespace.get("sample_keys", [])
        inherited_keys: List[str] = []
        for base in bases:
            inherited_keys.extend(getattr(base, "sample_keys", []))
        # Combine inherited and declared, preserving order and uniqueness
        keys_list: List[str] = []
        for k in inherited_keys + declared_keys:
            if k not in keys_list:
                keys_list.append(k)
        keys = keys_list

        # For each key, ensure there’s a get_<key>; if not, create an abstract stub
        if keys:
            for key in keys:
                method_name = f"get_{key}"
                if method_name not in namespace:
                    # define an abstractmethod stub
                    # define a stub that just raises NotImplementedError
                    def _stub(self, idx: int, _method_name=method_name):
                        raise NotImplementedError(
                            f"{name} must implement `{_method_name}`"
                        )

                    namespace[method_name] = abstractmethod(_stub)
        cls = super().__new__(mcls, name, bases, namespace)
        return cls


class BaseDataset(Dataset, metaclass=DatasetMeta):
    """
    Generic PyTorch Dataset with dynamic abstract getters.

    Attributes:
        sample_keys (List[str]): Keys defining each component returned by __getitem__.
    """

    sample_keys: List[str] = []

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve a sample by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict[str, torch.Tensor]: Mapping from each key in `sample_keys` to a tensor.
        """
        return {k: getattr(self, f"get_{k}")(idx) for k in self.sample_keys}

    def __init_subclass__(cls, **kwargs):
        """
        Enforce that all subclasses define a non-empty `sample_keys` attribute.
        """
        super().__init_subclass__(**kwargs)
        if cls is not BaseDataset and not getattr(cls, "sample_keys", None):
            raise TypeError(
                f"{cls.__name__} must define a non-empty `sample_keys` attribute"
            )

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """


# class MyContrastiveDataset(BaseDataset):
#     """
#     Example contrastive dataset.

#     Returns anchor, positive, and negative tensors for each sample index.
#     """

#     # sample_keys = ["anchor", "positive", "negative"]

#     def __len__(self) -> int:
#         return len(self.data_source)

#     def get_anchor(self, idx: int) -> torch.Tensor:
#         """
#         Get the anchor sample tensor for the given index.

#         Args:
#             idx (int): Sample index.

#         Returns:
#             torch.Tensor: Anchor tensor.
#         """
#         return torch.tensor(self.data_source[idx][0])

#     def get_positive(self, idx: int) -> torch.Tensor:
#         """
#         Get the positive sample tensor for the given index.

#         Args:
#             idx (int): Sample index.

#         Returns:
#             torch.Tensor: Positive tensor.
#         """
#         return torch.tensor(self.data_source[idx][1])

#     def get_negative(self, idx: int) -> torch.Tensor:
#         """
#         Get the negative sample tensor for the given index.

#         Args:
#             idx (int): Sample index.

#         Returns:
#             torch.Tensor: Negative tensor.
#         """
#         return torch.tensor(self.data_source[idx][2])
