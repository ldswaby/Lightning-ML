try:  # pragma: no cover - dependency check
    from pytorch_lightning.utilities.enums import LightningEnum
except Exception:  # pragma: no cover - fallback when lightning not installed
    from enum import Enum

    class LightningEnum(str, Enum):
        """Minimal fallback for :class:`pytorch_lightning.utilities.enums.LightningEnum`."""

        def __str__(self) -> str:  # pragma: no cover - parity with LightningEnum
            return str(self.value)


class DataKeys(LightningEnum):
    """The ``DataKeys`` enum contains the keys that are used by built-in data sources to refer to inputs and targets."""

    INPUT = "input"
    TARGET = "target"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    PREDS = "prediction"
    METADATA = "metadata"


class InputFormat(LightningEnum):
    """The ``InputFormat`` enum contains the data source names used by all of the default ``from_*`` methods in
    :class:`~flash.core.data.data_module.DataModule`."""

    FOLDERS = "folders"
    FILES = "files"
    NUMPY = "numpy"
    TENSORS = "tensors"
    CSV = "csv"
    # JSON = "json"
    # PARQUET = "parquet"
    # DATASETS = "datasets"
    # HUGGINGFACE_DATASET = "hf_datasets"
    # FIFTYONE = "fiftyone"
    # DATAFRAME = "data_frame"
    # LISTS = "lists"
    # LABELSTUDIO = "labelstudio"


class RunningStage(LightningEnum):
    """Enum for the current running stage.

    This stage complements :class:`TrainerFn` by specifying the current running stage for each function.
    More than one running stage value can be set while a :class:`TrainerFn` is running:

        - ``TrainerFn.FITTING`` - ``RunningStage.{SANITY_CHECKING,TRAINING,VALIDATING}``
        - ``TrainerFn.VALIDATING`` - ``RunningStage.VALIDATING``
        - ``TrainerFn.TESTING`` - ``RunningStage.TESTING``
        - ``TrainerFn.PREDICTING`` - ``RunningStage.PREDICTING``
        - ``TrainerFn.SERVING`` - ``RunningStage.SERVING``
        - ``TrainerFn.TUNING`` - ``RunningStage.{TUNING,SANITY_CHECKING,TRAINING,VALIDATING}``

    """

    TRAINING = "train"
    # SANITY_CHECKING = "sanity_check"
    VALIDATING = "validate"
    TESTING = "test"
    PREDICTING = "predict"
    # SERVING = "serve"
    # TUNING = "tune"


class Registries(LightningEnum):
    """The ``DataKeys`` enum contains the keys that are used by built-in data sources to refer to inputs and targets."""

    LOADER = "loader"
    DATASET = "dataset"
    MODEL = "model"
    LOSS = "loss"
    LEARNER = "learner"
    PREDICTOR = "predictor"
