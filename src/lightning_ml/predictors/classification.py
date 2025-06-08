from torch import Tensor

from ..core import Predictor

__all__ = ["Classification"]


class Classification(Predictor):
    """Classification predictor"""

    def __init__(self, softmax: bool = True) -> None:
        self.softmax = softmax

    def __call__(self, outputs: Tensor) -> Tensor:
        if self.softmax:
            outputs = outputs.softmax(dim=-1)
        return outputs.argmax(dim=-1)
