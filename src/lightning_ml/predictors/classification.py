from typing import Any, Dict

from torch import Tensor

from ..core import Predictor

__all__ = ["Classification"]


class Classification(Predictor):
    """Classification predictor"""

    def __call__(self, outputs: Dict[str, Any]) -> Tensor:
        return outputs["output"].softmax(dim=-1).argmax(dim=-1)
