from ..core import Predictor

__all__ = ["Regression"]


class Regression(Predictor):
    """Basic regression predictor"""

    def post_process(self, preds):
        return preds.squeeze(-1)
