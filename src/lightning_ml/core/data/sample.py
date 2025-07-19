from dataclasses import field, make_dataclass
from typing import Any, Optional

# wherever you defined your enum
from ..utils.enums import DataKeys

Sample = make_dataclass(
    "Sample",
    [(key.value, Optional[Any], field(default=None)) for key in DataKeys],
    frozen=True,
    kw_only=True,
    slots=True,
)
