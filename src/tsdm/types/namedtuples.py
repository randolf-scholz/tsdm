"""Named Tuples used by TSDM."""


__all__ = [
    "TableSchema",
]

from typing import NamedTuple, Sequence


class TableSchema(NamedTuple):
    """Table schema."""

    shape: tuple[int, int]
    """Shape of the table."""
    columns: Sequence[str]
    """Column names of the table."""
    dtypes: Sequence[str]
    """Data types of the columns."""
