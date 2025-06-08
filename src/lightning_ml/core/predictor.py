"""Predictor utilities and wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from .learner import Learner

__all__ = ["PredictorMixin", "PredictorWrapper"]


# class Predictor(ABC):
#     """Base class for prediction helpers."""

#     @abstractmethod
#     def post_process(self, outputs: Any) -> Any:
#         """Convert raw learner outputs into final predictions."""

#     def predict_step(
#         self,
#         learner: Learner,
#         batch: Any,
#         batch_idx: int,
#         dataloader_idx: int = 0,
#     ) -> Any:
#         """Default prediction logic that delegates to the learner."""
#         out = learner.predict_step(batch, batch_idx, dataloader_idx)
#         return self.post_process(out)


# class PredictorWrapper(pl.LightningModule):
#     """Wraps a learner with a predictor to override ``predict_step``."""

#     def __init__(self, learner: Learner, predictor: Predictor) -> None:
#         super().__init__()
#         self.learner = learner
#         self.predictor = predictor

#     # Delegate attribute access to the learner
#     def __getattr__(self, item: str):
#         if item in {"learner", "predictor"}:
#             return super().__getattribute__(item)
#         return getattr(self.learner, item)

#     def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401
#         """Delegate ``forward`` to the wrapped learner."""
#         return self.learner(*args, **kwargs)

#     def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
#         return self.predictor.predict_step(
#             self.learner, batch, batch_idx, dataloader_idx
#         )


# class Predictor(ABC):
#     """Base class for prediction helpers."""

#     @abstractmethod
#     def post_process(self, outputs: Any) -> Any:
#         """Convert raw learner outputs into final predictions."""

#     def predict_step(
#         self,
#         learner: Learner,
#         batch: Any,
#         batch_idx: int,
#         dataloader_idx: int = 0,
#     ) -> Any:
#         """Default prediction logic that delegates to the learner."""
#         out = learner.predict_step(batch, batch_idx, dataloader_idx)
#         return self.post_process(out)


class PredictorMixin(pl.LightningModule, ABC):
    """Wraps a learner with a predictor to override ``predict_step``."""

    def __init__(self, learner: "Learner") -> None:
        super().__init__()
        self._learner = learner

    @abstractmethod
    def post_process(self, outputs: Any) -> Any:
        """Convert raw learner outputs into final predictions."""

    def predict_step(self, *args, **kwargs) -> Any:
        """Default prediction logic that delegates to the learner."""
        out = self._learner.predict_step(*args, **kwargs)
        return self.post_process(out)

    def __getattr__(self, name):  # delegate attrs / hooks
        return getattr(self._learner, name)
