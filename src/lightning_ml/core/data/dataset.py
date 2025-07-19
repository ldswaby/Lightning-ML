"""
Module for abstract dataset definitions with dynamic abstract methods.

This module defines:
- AutoAbstractMeta: metaclass that auto-generates abstract stubs based on sample_keys.
- BaseDataset: a generic PyTorch Dataset using dynamic getters.
- MyContrastiveDataset: example subclass for contrastive learning paradigms.
"""

from abc import ABCMeta, abstractmethod

# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from ..utils.enums import DataKeys
from .sample import Sample

__all__ = [
    "BaseDataset",
    "InputMixin",
    "TargetMixin",
    "PositiveMixin",
    "NegativeMixin",
    "UnlabelledDatasetBase",
    "LabelledDatasetBase",
    "ContastiveDatasetBase",
    "TripletDatasetBase",
]


class DatasetMeta(ABCMeta):
    """
    Metaclass that auto-injects abstract getter stubs for each key in `sample_keys`.

    If a subclass defines `sample_keys`, any missing `get_<key>` methods
    will be created as abstract stubs that raise NotImplementedError.

    NOTE: This enables further abtsract Dataset classes to be created simply from list of required sample keys
    """

    def __new__(
        mcls, name: str, bases: tuple[type, ...], namespace: dict[str, Any]
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
        declared_keys: list[str] = namespace.get("sample_keys", [])
        inherited_keys: list[str] = []
        for base in bases:
            inherited_keys.extend(getattr(base, "sample_keys", []))
        # Combine inherited and declared, preserving order and uniqueness
        keys_list: list[str] = []
        for k in inherited_keys + declared_keys:
            if k not in keys_list:
                keys_list.append(k)
        keys = keys_list
        # ensure the resulting class exposes the aggregated keys
        namespace["sample_keys"] = keys

        # For each key, ensure there’s a get_<key>; if not, create an abstract
        # stub
        if keys:
            for key in keys:
                method_name = f"get_{key}"
                if method_name not in namespace and not any(
                    hasattr(base, method_name) for base in bases
                ):
                    # define an abstractmethod stub
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

    sample_keys: list[str] = []

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Retrieve a sample by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict[str, torch.Tensor]: Mapping from each key in `sample_keys` to a tensor.
        """
        return Sample(**{k: getattr(self, f"get_{k}")(idx) for k in self.sample_keys})

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


class InputMixin(BaseDataset):
    """Dataset mix-in that mandates a `get_input` method, the output of which
    will be added to the `__getitem__` ouput dict of all subclasses with key
    "input".
    """

    sample_keys = [DataKeys.INPUT]


class TargetMixin(BaseDataset):
    """Dataset mix-in that mandates a `get_target` method."""

    sample_keys = [DataKeys.TARGET]


class PositiveMixin(BaseDataset):
    """Dataset mix-in that mandates a `get_positive` method."""

    sample_keys = [DataKeys.POSITIVE]


class NegativeMixin(BaseDataset):
    """Dataset mix-in that mandates a `get_negative` method."""

    sample_keys = [DataKeys.NEGATIVE]


# Abstract dataset bases


class UnlabelledDatasetBase(InputMixin):
    """
    Abstract base for unlabeled datasets.

    Combines mix-ins to enforce retrieval of input samples only.
    """


class LabelledDatasetBase(InputMixin, TargetMixin):
    """
    Abstract base for labeled datasets.

    Combines mix-ins to enforce retrieval of input and target samples.
    """


class ContastiveDatasetBase(InputMixin, PositiveMixin):
    """
    Abstract base for contrastive datasets.

    Combines mix-ins to enforce retrieval of input and positive samples.
    """


class TripletDatasetBase(ContastiveDatasetBase, NegativeMixin):
    """
    Abstract base for triplet datasets.

    Combines mix-ins to enforce retrieval of input, positive, and negative samples.
    """
