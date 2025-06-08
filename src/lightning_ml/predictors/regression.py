from ..core import PredictorMixin

__all__ = ["Regression"]


class Regression(PredictorMixin):
    """Basic regression predictor"""

    def post_process(self, preds):
        return preds.squeeze(-1)
