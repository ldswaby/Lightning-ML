from pytorch_lightning.utilities.enums import LightningEnum


class DataKeys(LightningEnum):
    """The ``DataKeys`` enum contains the keys that are used by built-in data sources to refer to inputs and targets."""

    INPUT = "input"
    PREDS = "preds"
    TARGET = "target"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    METADATA = "metadata"

    def __hash__(self) -> int:
        return hash(self.value)


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

    def __hash__(self) -> int:
        return hash(self.value)


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
