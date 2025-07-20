from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from lightning_ml.core.data import BaseLoader
from lightning_ml.core.utils.imports import requires


@requires("pandas")
@dataclass
class CSVLoader(BaseLoader):
    """Loads tabular data from a CSV file.

    Args:
        path: Path to the ``.csv`` file.
        input_cols: Column names that form the model inputs.
        target_cols: Column names that form the targets.  If *None*, no targets
            are returned.
        read_kwargs: Additional keyword arguments forwarded to
            :pyfunc:`pandas.read_csv`.
    """

    path: str
    input_cols: Sequence[str]
    target_cols: Optional[Sequence[str]] = None
    read_kwargs: Optional[Dict[str, Any]] = None

    @property
    def _df(self) -> "pd.DataFrame":
        import pandas as pd

        return pd.read_csv(self.path, **(self.read_kwargs or {}))

    def inputs(self) -> Sequence:
        return self._df[list(self.input_cols)].to_dict("records")

    def targets(self) -> Optional[Sequence]:
        if self.target_cols is None:
            return None
        return self._df[list(self.target_cols)].to_dict("records")
