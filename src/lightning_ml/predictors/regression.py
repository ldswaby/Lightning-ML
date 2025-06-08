from typing import Any, Dict

from torch import Tensor

from ..core import Predictor

__all__ = ["Regression"]


class Regression(Predictor):
    """Basic regression predictor"""

    def __call__(self, outputs: Dict[str, Any]) -> Tensor:
        return outputs["output"].squeeze(-1)
