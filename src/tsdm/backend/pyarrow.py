r"""Implements `pyarrow`-backend for tsdm."""

__all__ = [
    # Constants
    "STR",
    "TEXT",
    "STRING_TYPES",
    # Functions
    "and_",
    "cast_column",
    "cast_columns",
    "compute_entropy",
    "false_like",
    "filter_nulls",
    "force_cast",
    "full_like",
    "is_numeric",
    "is_string_array",
    "null_like",
    "or_",
    "scalar",
    "set_nulls",
    "set_nulls_series",
    "strip_whitespace",
    "table_info",
    "true_like",
    "unsafe_cast_columns",
    "where",
    # auxiliary Functions
    "strip_whitespace_table",
    "strip_whitespace_array",
]

from collections.abc import Sequence
from typing import Literal, overload

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import (
    NA,
    Array,
    BooleanArray,
    BooleanScalar,
    DataType,
    DictionaryArray,
    ListArray,
    Scalar,
    Table,
)
from tqdm import tqdm

from tsdm.types.dtypes import PYARROW_TO_POLARS

STR = pa.string()
TEXT = pa.large_string()
STRING_TYPES = frozenset({STR, TEXT})


def scalar(x: object, /, dtype: DataType) -> Scalar:
    return pa.scalar(x, type=dtype)


def is_string_array(arr: Array, /) -> bool:
    r"""Check if an array is a string array."""
    match arr.type:
        case _ if arr.type in STRING_TYPES:
            return True
        case pa.ListType(value_type=value_type):
            return value_type in STRING_TYPES  # type: ignore[has-type]
        case pa.DictionaryType(value_type=value_type):
            return value_type in STRING_TYPES  # type: ignore[has-type]
        case _:
            return False


def strip_whitespace_table(table: Table, /, *cols: str) -> Table:
    r"""Strip whitespace from selected columns in table."""
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
    r"""Strip whitespace from all string elements in an array."""
    match arr:
        case pa.ChunkedArray(chunks=chunks):
            return pa.chunked_array(map(strip_whitespace_array, chunks))  # type: ignore[has-type]
        case ListArray(type=dtype) if dtype.value_type in STRING_TYPES:
            return pa.array(map(pc.utf8_trim_whitespace, arr), type=dtype)
        case DictionaryArray(
            type=dtype, indices=indices, dictionary=dictionary
        ) if dtype.value_type in STRING_TYPES:
            return DictionaryArray.from_arrays(
                indices,
                pc.utf8_trim_whitespace(dictionary),
            )
        case Array(type=dtype) if dtype in STRING_TYPES:
            return pc.utf8_trim_whitespace(arr)
        case _:
            raise TypeError(f"Expected string array, got {arr.type}.")


@overload
def strip_whitespace(obj: Table, /, *cols: str) -> Table: ...
@overload
def strip_whitespace(obj: Array, /, *cols: str) -> Array: ...  # type: ignore[misc]
def strip_whitespace(obj, /, *cols):
    r"""Strip whitespace from all string elements in an arrow object."""
    match obj:
        case Table() as table:
            return strip_whitespace_table(table, *cols)
        case Array() as array:
            if cols:
                raise ValueError("Cannot specify columns for an Array.")
            return strip_whitespace_array(array)
        case _:
            raise TypeError(f"Expected Array or Table, got {type(obj)}.")


def false_like(arr: Array, /) -> BooleanArray:
    r"""Creates a `BooleanArray` of False values with the same length as arr."""
    m = arr.is_valid()
    return pc.xor(m, m)


def true_like(arr: Array, /) -> BooleanArray:
    r"""Creates a `BooleanArray` of True values with the same length as arr."""
    return pc.invert(false_like(arr))


def full_like(arr: Array, /, *, fill_value: Scalar) -> Array:
    r"""Creates an `Array` of `fill_value` with the same length as arr."""
    if not isinstance(fill_value, Scalar):
        fill_value = pa.scalar(fill_value)
    if fill_value is NA:
        fill_value = fill_value.cast(arr.type)
    if fill_value.type == arr.type:
        return pc.replace_with_mask(arr, false_like(arr), fill_value)
    empty = null_like(arr).cast(fill_value.type)
    return pc.replace_with_mask(empty, true_like(arr), fill_value)


def null_like(arr: Array, /) -> Array:
    r"""Creates an `Array` of null-values with the same length as arr."""
    return full_like(arr, fill_value=NA)


@overload
def where(mask: BooleanScalar, x: Scalar, y: Scalar = ..., /) -> Scalar: ...
@overload
def where(  # type: ignore[misc]
    mask: BooleanArray | BooleanScalar,
    x: Array | Scalar,
    y: Array | Scalar = ...,
    /,
) -> Array: ...
def where(mask, x, y=NA, /):
    r"""Select elements from x or y depending on mask.

    arrow_where(mask, x, y) is roughly equivalent to x.where(mask, y).
    """
    return pc.replace_with_mask(x, mask, y)


@overload
def force_cast(x: Table, dtype: DataType, /) -> Table: ...
@overload
def force_cast(x: Array, dtype: DataType, /) -> Array: ...  # type: ignore[misc]
@overload
def force_cast(x: Table, /, **dtypes: DataType) -> Table: ...
def force_cast(x, dtype=None, /, **dtypes):
    r"""Cast an array or table to the given data type, replacing non-castable elements with null."""
    match x:
        case Array() as array:
            array = array.combine_chunks()  # deals with chunked arrays
            if dtype is None:
                raise ValueError("Must specify dtype for Array input.")
            if dtypes:
                raise ValueError("Unexpected argument dtypes for Array input.")

            return (
                pl.from_arrow(array)
                .cast(PYARROW_TO_POLARS[dtype], strict=False)
                .to_arrow()
                .cast(dtype)
            )
        case Table() as table:
            if unknown_keys := set(dtypes.keys()) - set(table.column_names):
                raise ValueError(f"Keys: {unknown_keys} not in table columns.")

            schema: pa.Schema = table.schema
            current_dtypes = dict(zip(schema.names, schema.types, strict=True))
            new_schema = pa.schema(current_dtypes | dtypes)

            return pa.table({
                name: force_cast(table[name], dtypes.get(name) or current_dtypes[name])
                for name in table.column_names
            }).cast(new_schema)
        case _:
            raise TypeError(f"Expected Array or Table, got {type(x)}.")


def cast_column(table: Table, col: str, dtype: DataType, /, *, safe: bool) -> Table:
    r"""Concatenate columns into a new column."""
    index = table.column_names.index(col)
    try:
        casted_column = (
            table[col].combine_chunks().dictionary_encode()
            if isinstance(dtype, pa.DictionaryType)
            else table[col].cast(dtype, safe=safe)
        )
    except Exception as exc:
        raise RuntimeError(
            f"Error {exc!r} occurred while casting column {col!r} to {dtype!r}."
        ) from exc
    return table.set_column(index, col, casted_column)


def cast_columns(table: Table, /, **dtypes: DataType) -> Table:
    r"""Cast columns to the given data types."""
    schema: pa.Schema = table.schema
    current_dtypes = dict(zip(schema.names, schema.types, strict=True))
    if unknown_keys := set(dtypes.keys()) - set(current_dtypes.keys()):
        raise ValueError(f"Keys: {unknown_keys} not in table columns.")

    new_dtypes = current_dtypes | dtypes

    for col, dtype in new_dtypes.items():
        table = cast_column(table, col, dtype, safe=True)
    return table


def unsafe_cast_columns(table: Table, /, **dtypes: DataType) -> Table:
    r"""Cast columns to the given data types, replacing non-castable elements with null."""
    schema: pa.Schema = table.schema
    current_dtypes = dict(zip(schema.names, schema.types, strict=True))
    if unknown_keys := set(dtypes.keys()) - set(current_dtypes.keys()):
        raise ValueError(f"Keys: {unknown_keys} not in table columns.")

    new_dtypes = current_dtypes | dtypes

    for col, dtype in new_dtypes.items():
        table = cast_column(table, col, dtype, safe=False)
    return table


def is_numeric(array: Array, /) -> Array:
    r"""Return mask determining if each element can be cast to the given data type."""
    prior_null = pc.is_null(array)
    post_null = pc.is_null(
        Array.from_pandas(
            pd.to_numeric(
                pd.Series(array, dtype="string[pyarrow]"),
                # NOTE: string[pyarrow] is actually `StringDtype`, so we can't use `to_pandas`
                errors="coerce",
                dtype_backend="pyarrow",
                downcast="float",
            )
        )
    )

    return pc.or_(
        prior_null,
        pc.invert(post_null),
    )


def compute_entropy(value_counts: Array, /) -> float:
    r"""Compute the normalized entropy using a value_counts array.

    .. math:: ∑_{i=1}^n -pᵢ \log₂(pᵢ)/\log₂(n)

    Note:
        Since entropy is maximized for a uniform distribution, and the entropy
        of a uniform distribution of n choices is log₂(n), the normalization
        ensures that the entropy is in the range [0, 1].
    """
    counts = pc.struct_field(value_counts, 1)

    freqs = pc.divide(
        pc.cast(counts, pa.float64()),
        pc.sum(counts),
    )

    H = pc.divide(
        pc.sum(pc.multiply(freqs, pc.log2(freqs))),
        pc.log2(len(counts)),
    )
    return -H.as_py()


def or_(masks: Sequence[Array], /) -> Array:
    r"""Compute the logical OR of a sequence of boolean arrays."""
    match n := len(masks):
        case 0:
            return pa.array([])
        case 1:
            return masks[0]
        case _:
            return pc.or_(or_(masks[: n // 2]), or_(masks[n // 2 :]))


def and_(masks: Sequence[Array], /) -> Array:
    r"""Compute the logical AND of a sequence of boolean arrays."""
    match n := len(masks):
        case 0:
            return pa.array([])
        case 1:
            return masks[0]
        case _:
            return pc.and_(and_(masks[: n // 2]), and_(masks[n // 2 :]))


def filter_nulls(
    table: Table, /, *cols: str, aggregation: Literal["or", "and"] = "or"
) -> Table:
    r"""Filter rows with null values in the given columns."""
    agg = {"or": or_, "and": and_}[aggregation]
    masks = [table[col].is_null() for col in cols]
    mask = pc.invert(agg(masks))
    return table.filter(mask)


def set_nulls_series(series: Array, values: Sequence, /) -> Array:
    r"""Set values to null if they match any of the given values."""
    mask = pc.is_in(series, pa.array(values, type=series.type))
    return pc.replace_with_mask(series, mask, pa.null())


def set_nulls(table: Table, /, **cols: Sequence) -> Table:
    r"""For given columns set all matching values ito null."""
    for col, values in cols.items():
        table = table.set_column(col, set_nulls_series(table[col], values))

    return table


def table_info(table: Table, /) -> None:
    r"""Print information about a table."""
    size = table.nbytes / (1024 * 1024 * 1024)
    print(f"shape={table.shape}  {size=:.3f} GiB")
    max_keylen = max(map(len, table.column_names)) + 1
    for name, col in tqdm(zip(table.column_names, table.columns, strict=True)):
        num_total = len(col)
        num_null = pc.sum(pc.is_null(col)).as_py()
        value_counts = col.value_counts()
        num_uniques = len(value_counts) - bool(num_null)
        nulls = f"{num_null / num_total:8.3%}" if num_null else "--------"
        uniques = (
            num_uniques / (num_total - num_null)
            if num_total > num_null
            else num_uniques / num_total
        )
        entropy = compute_entropy(value_counts)
        dtype = str(col.type)[:10]
        print(
            f"{name:{max_keylen}s}  {nulls=:s}  {num_uniques=:9d} ({uniques:8.3%})"
            f"  {entropy=:8.3%}  {dtype=:s}"
        )
