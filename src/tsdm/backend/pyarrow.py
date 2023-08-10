"""implement pyarrow backend for tsdm."""

__all__ = [
    # Functions
    "is_string_array",
    "arrow_strip_whitespace",
    # auxiliary Functions
    "strip_whitespace_table",
    "strip_whitespace_array",
]
import logging
from typing import TypeVar

import pyarrow as pa

__logger__ = logging.getLogger(__name__)


P = TypeVar("P", pa.Array, pa.Table)
"""A type variable for pyarrow objects."""


def is_string_array(arr: pa.Array) -> bool:
    """Check if an array is a string array."""
    if arr.type in (pa.string(), pa.large_string()):
        return True
    if isinstance(arr.type, pa.ListType) and arr.type.value_type in (
        pa.string(),
        pa.large_string(),
    ):
        return True
    if isinstance(arr.type, pa.DictionaryType) and arr.type.value_type in (
        pa.string(),
        pa.large_string(),
    ):
        return True
    return False


def strip_whitespace_table(table: pa.Table, *cols: str) -> pa.Table:
    """Strip whitespace from all string columns in a table."""
    for col in cols or table.column_names:
        if is_string_array(table[col]):
            __logger__.debug("Trimming the string column %r", col)
            table = table.set_column(
                table.column_names.index(col),
                col,
                strip_whitespace_array(table[col]),
            )
        else:
            __logger__.debug("Ignoring non string column %r", col)
    return table


def strip_whitespace_array(arr: pa.Array) -> pa.Array:
    """Strip whitespace from all string elements in an array."""
    if isinstance(arr, pa.ChunkedArray):
        return pa.chunked_array(
            [strip_whitespace_array(chunk) for chunk in arr.chunks],
        )
    if isinstance(arr, pa.Array) and arr.type == pa.string():
        return pa.compute.utf8_trim_whitespace(arr)
    if isinstance(arr, pa.ListArray) and arr.type.value_type == pa.string():
        return pa.compute.map(
            pa.compute.utf8_trim_whitespace,
            arr,
        )
    if isinstance(arr, pa.DictionaryArray) and arr.type.value_type == pa.string():
        return pa.DictionaryArray.from_arrays(
            arr.indices,
            pa.compute.utf8_trim_whitespace(arr.dictionary),
        )
    raise ValueError(f"Expected string array, got {arr.type}.")


def arrow_strip_whitespace(obj: P) -> P:
    """Strip whitespace from all string elements in an arrow object."""
    if isinstance(obj, pa.Table):
        return strip_whitespace_table(obj)
    if isinstance(obj, pa.Array):
        return strip_whitespace_array(obj)
    raise TypeError(f"Expected pa.Array or pa.Table, got {type(obj)}.")
