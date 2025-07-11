# # TODO: prunt this
# from typing import Optional

# from ..utils.enums import RunningStage


# class Properties:
#     def __init__(
#         self,
#         running_stage: Optional[RunningStage] = None,
#     ):
#         super().__init__()

#         self._running_stage = running_stage

#     @property
#     def running_stage(self) -> Optional[RunningStage]:
#         return self._running_stage

#     @property
#     def training(self) -> bool:
#         return self._running_stage == RunningStage.TRAINING

#     @training.setter
#     def training(self, val: bool) -> None:
#         if val:
#             self._running_stage = RunningStage.TRAINING
#         elif self.training:
#             self._running_stage = None

#     @property
#     def validating(self) -> bool:
#         return self._running_stage == RunningStage.VALIDATING

#     @validating.setter
#     def validating(self, val: bool) -> None:
#         if val:
#             self._running_stage = RunningStage.VALIDATING
#         elif self.validating:
#             self._running_stage = None

#     @property
#     def testing(self) -> bool:
#         return self._running_stage == RunningStage.TESTING

#     @testing.setter
#     def testing(self, val: bool) -> None:
#         if val:
#             self._running_stage = RunningStage.TESTING
#         elif self.testing:
#             self._running_stage = None

#     @property
#     def predicting(self) -> bool:
#         return self._running_stage == RunningStage.PREDICTING

#     @predicting.setter
#     def predicting(self, val: bool) -> None:
#         if val:
#             self._running_stage = RunningStage.PREDICTING
#         elif self.predicting:
#             self._running_stage = None
