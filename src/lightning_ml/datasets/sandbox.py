"""
Module for abstract dataset definitions with dynamic abstract methods.

This module defines:
- AutoAbstractMeta: metaclass that auto-generates abstract stubs based on sample_keys.
- BaseDataset: a generic PyTorch Dataset using dynamic getters.
- Mixin classes for anchor/positive/negative sample keys.
- Labelled and unlabelled dataset bases.
- LabelledDataset: in-memory labelled dataset.
- ContrastiveLabelledDataset: returns anchor and positive (same label).
- UnlabelledContrastiveDataset: returns two augmented views of a sample.
- TripletLabelledDataset: returns (anchor, positive, negative) triplets.
"""

import random
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

__all__ = [
    "BaseDataset",
    "UnlabelledDatasetBase",
    "LabelledDatasetBase",
    "LabelledDataset",
    "ContrastiveLabelledDataset",
    "UnlabelledContrastiveDataset",
    "TripletLabelledDataset",
]


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
        # Create the class
        cls = super().__new__(mcls, name, bases, namespace)
        # Enforce that subclasses (except BaseDataset itself) define sample_keys
        if not getattr(cls, "sample_keys", None):
            raise TypeError(
                f"{name} must define a non-empty `sample_keys` attribute. "
                f"Please see www.google.com."  # TODO: update URL
            )
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

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """


##### mixins that enforce subclasses to define method get_{name} for name in self.sample_keys


class InputMixin(BaseDataset):
    """Dataset mix-in that mandates a `get_input` method."""

    sample_keys = ["input"]


class TargetMixin(BaseDataset):
    """Dataset mix-in that mandates a `get_target` method."""

    sample_keys = ["target"]


class PositiveMixin(BaseDataset):
    """Dataset mix-in that mandates a `get_positive` method."""

    sample_keys = ["positive"]


class NegativeMixin(BaseDataset):
    """Dataset mix‑in that mandates a `get_negative` method."""

    sample_keys = ["negative"]


##### abstract datasets
class UnlabelledMixin(InputMixin):
    pass


class LabelledMixin(InputMixin, TargetMixin):
    pass


class ContastiveMixin(InputMixin, PositiveMixin):
    pass


class TripletMixin(ContastiveMixin, NegativeMixin):
    pass


class LabelledDataset(LabelledMixin):
    """
    In-memory labeled dataset.
    """

    def __init__(self, inputs: Sequence[Any], targets: Sequence[Any]) -> None:
        self._inputs = inputs
        self._targets = targets

    def __len__(self) -> int:
        """
        Return the number of samples.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self._targets)

    def get_input(self, idx: int) -> Any:
        """
        Retrieve the input sample at the given index.

        Args:
            idx (int): index of the sample.

        Returns:
            Any: Input data at the specified index.
        """
        return self._inputs[idx]

    def get_target(self, idx: int) -> Any:
        """
        Retrieve the target at the given index.

        Args:
            idx (int): Index of the target.

        Returns:
            Any: Target data at the specified index.
        """
        return self._targets[idx]


class ContrastiveLabelledDataset(ContastiveMixin, LabelledDataset):
    """
    In-memory contrastive dataset for labelled data.

    Each sample returns an anchor and a positive sample with the same label.
    """

    def __init__(self, inputs: Sequence[Any], targets: Sequence[Any]) -> None:
        """
        Initialize the contrastive dataset.

        Args:
            inputs (Sequence[Any]): Input data sequence.
            targets (Sequence[Any]): Target labels corresponding to inputs.
        """
        super().__init__(inputs, targets)
        # Build mapping from label to list of indices
        self._label_to_idxs = {}
        for idx, label in enumerate(self._targets):
            self._label_to_idxs.setdefault(label, []).append(idx)

    def get_positive(self, idx: int) -> Any:
        """
        Retrieve a positive sample (same label as anchor).
        Falls back to anchor if no other positive exists.
        """
        label = self._targets[idx]
        pos_idxs = self._label_to_idxs[label]
        if len(pos_idxs) < 2:
            return self._inputs[idx]
        pos_idx = idx
        while pos_idx == idx:
            pos_idx = random.choice(pos_idxs)
        return self._inputs[pos_idx]


# Unlabelled contrastive dataset: two augmented views of same sample


class UnlabelledContrastiveDataset(ContastiveMixin):
    """
    Contrastive dataset without labels.

    Each call returns two stochastically augmented views of the same sample.
    """

    def __init__(
        self,
        inputs: Sequence[Any],
        transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        """
        Args:
            inputs (Sequence[Any]): Raw input sequence.
            transform (Callable, optional): Optional augmentation callable applied
                independently to anchor and positive views. If ``None`` the raw
                sample is returned (identity transform).
        """
        self._inputs = inputs
        self.transform = transform or (lambda x: x)

    def __len__(self) -> int:
        return len(self._inputs)

    def get_input(self, idx: int) -> Any:
        """Return the (possibly augmented) anchor view."""
        return self.transform(self._inputs[idx])

    def get_positive(self, idx: int) -> Any:
        """Return an independent augmented view of the same sample."""
        return self.transform(self._inputs[idx])


# Triplet dataset: anchor, positive, negative


class TripletLabelledDataset(TripletMixin, ContrastiveLabelledDataset):
    """
    Triplet dataset returning (anchor, positive, negative) given class labels.
    """

    def get_negative(self, idx: int) -> Any:
        anchor_label = self._targets[idx]
        neg_label = anchor_label
        while neg_label == anchor_label:
            neg_label = random.choice(list(self._label_to_idxs.keys()))
        neg_idx = random.choice(self._label_to_idxs[neg_label])
        return self._inputs[neg_idx]
