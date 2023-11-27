"""Implements `pyarrow`-backend for tsdm."""

__all__ = [
    # Functions
    "is_string_array",
    "arrow_strip_whitespace",
    "arrow_false_like",
    "arrow_true_like",
    "arrow_null_like",
    # auxiliary Functions
    "strip_whitespace_table",
    "strip_whitespace_array",
]


import pyarrow as pa
from pyarrow import (
    NA,
    Array,
    BooleanArray,
    BooleanScalar,
    DictionaryArray,
    ListArray,
    Scalar,
    Table,
)
from typing_extensions import overload


def is_string_array(arr: Array, /) -> bool:
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


def strip_whitespace_table(table: Table, /, *cols: str) -> Table:
    """Strip whitespace from selected columns in table."""
    for col in cols or table.column_names:
        if is_string_array(table[col]):
            # Trimming the string column col
            table = table.set_column(
                table.column_names.index(col),
                col,
                strip_whitespace_array(table[col]),
            )
    return table


def strip_whitespace_array(arr: Array, /) -> Array:
    """Strip whitespace from all string elements in an array."""
    if isinstance(arr, pa.ChunkedArray):
        return pa.chunked_array(
            [strip_whitespace_array(chunk) for chunk in arr.chunks],
        )
    if isinstance(arr, ListArray) and arr.type.value_type == pa.string():
        return pa.compute.map(
            pa.compute.utf8_trim_whitespace,
            arr,
        )
    if isinstance(arr, DictionaryArray) and arr.type.value_type == pa.string():
        return DictionaryArray.from_arrays(
            arr.indices,
            pa.compute.utf8_trim_whitespace(arr.dictionary),
        )
    if isinstance(arr, Array) and arr.type in {pa.string(), pa.large_string()}:
        return pa.compute.utf8_trim_whitespace(arr)
    raise ValueError(f"Expected string array, got {arr.type}.")


@overload
def arrow_strip_whitespace(obj: Table, /, *cols: str) -> Table: ...
@overload
def arrow_strip_whitespace(obj: Array, /, *cols: str) -> Array: ...
def arrow_strip_whitespace(obj, /, *cols):
    """Strip whitespace from all string elements in an arrow object."""
    if isinstance(obj, Table):
        return strip_whitespace_table(obj, *cols)
    if isinstance(obj, Array):
        if cols:
            raise ValueError("Cannot specify columns for an Array.")
        return strip_whitespace_array(obj)
    raise TypeError(f"Expected Array or Table, got {type(obj)}.")


def arrow_false_like(arr: Array, /) -> BooleanArray:
    """Creates a `BooleanArray` of False values with the same length as arr."""
    m = arr.is_valid()
    return pa.compute.xor(m, m)


def arrow_true_like(arr: Array, /) -> BooleanArray:
    """Creates a `BooleanArray` of True values with the same length as arr."""
    return pa.compute.invert(arrow_false_like(arr))


def arrow_full_like(arr: Array, /, *, fill_value: Scalar) -> Array:
    """Creates an `Array` of `fill_value` with the same length as arr."""
    if not isinstance(fill_value, Scalar):
        fill_value = pa.scalar(fill_value)
    if fill_value is NA:
        fill_value = fill_value.cast(arr.type)
    if fill_value.type == arr.type:
        return pa.compute.replace_with_mask(arr, arrow_false_like(arr), fill_value)
    empty = arrow_null_like(arr).cast(fill_value.type)
    return pa.compute.replace_with_mask(empty, arrow_true_like(arr), fill_value)


def arrow_null_like(arr: Array, /) -> Array:
    """Creates an `Array` of null-values with the same length as arr."""
    return arrow_full_like(arr, fill_value=NA)


@overload
def arrow_where(mask: BooleanScalar, x: Scalar, y: Scalar = NA, /) -> Scalar: ...
@overload
def arrow_where(  # type: ignore[misc]
    mask: BooleanArray | BooleanScalar,
    x: Array | Scalar,
    y: Array | Scalar = NA,
    /,
) -> Array: ...
def arrow_where(mask, x, y=NA, /):
    """Select elements from x or y depending on mask.

    arrow_where(mask, x, y) is roughly equivalent to x.where(mask, y).
    """
    pa.compute.replace_with_mask(x, mask, y)
