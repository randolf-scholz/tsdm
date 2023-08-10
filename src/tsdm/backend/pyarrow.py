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
from typing import TypeVar, overload

import pyarrow as pa

__logger__ = logging.getLogger(__name__)


P = TypeVar("P", pa.Array, pa.Table)
"""A type variable for pyarrow objects."""


def is_string_array(arr: pa.Array, /) -> bool:
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


def strip_whitespace_table(table: pa.Table, /, *cols: str) -> pa.Table:
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


def strip_whitespace_array(arr: pa.Array, /) -> pa.Array:
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


def arrow_strip_whitespace(obj: P, /) -> P:
    """Strip whitespace from all string elements in an arrow object."""
    if isinstance(obj, pa.Table):
        return strip_whitespace_table(obj)
    if isinstance(obj, pa.Array):
        return strip_whitespace_array(obj)
    raise TypeError(f"Expected pa.Array or pa.Table, got {type(obj)}.")


def arrow_false_like(arr: pa.Array, /) -> pa.BooleanArray:
    """Create an BooleanArray of False values with the same length as arr."""
    m = arr.is_valid()
    return pa.compute.xor(m, m)


def arrow_true_like(arr: pa.Array, /) -> pa.BooleanArray:
    """Create an BooleanArray of True values with same length as arr."""
    return pa.compute.invert(arrow_false_like(arr))


def arrow_full_like(arr: pa.Array, /, fill_value: pa.Scalar) -> pa.Array:
    """Create an Array of fill_value with same length as arr."""
    if not isinstance(fill_value, pa.Scalar):
        fill_value = pa.scalar(fill_value)
    if fill_value is pa.NA:
        fill_value = fill_value.cast(arr.type)
    if fill_value.type == arr.type:
        return pa.compute.replace_with_mask(arr, arrow_false_like(arr), fill_value)
    empty = arrow_nan_like(arr).cast(fill_value.type)
    return pa.compute.replace_with_mask(empty, arrow_true_like(arr), fill_value)


def arrow_nan_like(arr: pa.Array, /) -> pa.Array:
    """Create an Array of NaN values with same length as arr."""
    return arrow_full_like(arr, pa.NA)


@overload
def arrow_where(
    mask: pa.BooleanScalar, x: pa.Scalar, y: pa.Scalar = pa.NA
) -> pa.Scalar:
    ...


@overload
def arrow_where(
    mask: pa.BooleanArray | pa.BooleanScalar,
    x: pa.Array | pa.Scalar,
    y: pa.Array | pa.Scalar = pa.NA,
) -> pa.Array:
    ...


def arrow_where(mask, x, y=pa.NA):
    """Select elements from x or y depending on mask.

    arrow_where(mask, x, y) is roughly equivalent to x.where(mask, y).
    """
    pa.compute.replace_with_mask(x, mask, y)
