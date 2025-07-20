from dataclasses import asdict, field, make_dataclass
from typing import Any, Optional

# wherever you defined your enum
from ..utils.enums import DataKeys


def to_dict(self) -> dict[str, Any]:
    """Return a dictionary representation of the Sample instance."""
    return asdict(self)


Sample = make_dataclass(
    "Sample",
    [(key.value, Optional[Any], field(default=None)) for key in DataKeys],
    frozen=True,
    kw_only=True,
    slots=True,
    namespace={"to_dict": to_dict},
)
