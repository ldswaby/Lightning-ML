from torch import Tensor

from ..core import Predictor

__all__ = ["Regression"]


class Regression(Predictor):
    """Basic regression predictor"""

    def __call__(self, outputs: Tensor) -> Tensor:
        return outputs.squeeze(-1)
