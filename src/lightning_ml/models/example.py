"""Example model registered for demonstration purposes."""

from __future__ import annotations

import torch.nn as nn

from ..core.utils.enums import Registries
from ..core.utils.registry import register

__all__ = ["MyCustomModel"]


@register(Registries.MODEL, "MyCustomModel")
class MyCustomModel(nn.Module):
    """Simple fully connected network used as an example."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
