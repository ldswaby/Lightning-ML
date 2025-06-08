from torch import Tensor

from ..core import Predictor

__all__ = ["Classification"]


class Classification(Predictor):
    """Classification predictor"""

    def post_process(self, logits: Tensor) -> Tensor:
        return logits.softmax(dim=-1).argmax(dim=-1)
