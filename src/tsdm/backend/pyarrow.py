"""implement pyarrow backend for tsdm."""

__all__ = [
    # Functions
    "arrow_strip_whitespace",
    # auxiliary Functions
    "strip_whitespace_table",
    "strip_whitespace_array",
]

from typing import TypeVar

import pyarrow as pa

P = TypeVar("P", pa.Array, pa.Table)
"""A type variable for pyarrow objects."""


def strip_whitespace_table(table: pa.Table) -> pa.Table:
    """Strip whitespace from all string columns in a table."""
    for col in table.column_names:
        table = table.set_column(
            table.column_names.index(col),
            col,
            strip_whitespace_array(table[col]),
        )
    return table


def strip_whitespace_array(arr: pa.Array) -> pa.Array:
    """Strip whitespace from all string elements in an array."""
    if isinstance(arr, pa.Array) and arr.type == pa.string():
        return pa.compute.strip(arr)
    if isinstance(arr, pa.ListArray) and arr.type.value_type == pa.string():
        return pa.compute.map(
            pa.compute.strip,
            arr,
        )
    if isinstance(arr, pa.DictionaryArray) and arr.type.value_type == pa.string():
        return pa.DictionaryArray.from_arrays(
            arr.indices,
            pa.compute.strip(arr.dictionary),
        )
    return arr


def arrow_strip_whitespace(obj: P) -> P:
    """Strip whitespace from all string elements in an arrow object."""
    if isinstance(obj, pa.Table):
        return strip_whitespace_table(obj)
    if isinstance(obj, pa.Array):
        return strip_whitespace_array(obj)
    raise TypeError(f"Expected pa.Array or pa.Table, got {type(obj)}.")
