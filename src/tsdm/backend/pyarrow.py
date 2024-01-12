"""Implements `pyarrow`-backend for tsdm."""

__all__ = [
    # Functions
    "is_string_array",
    "arrow_strip_whitespace",
    "arrow_false_like",
    "arrow_true_like",
    "arrow_null_like",
    "force_cast",
    "cast_columns",
    "cast_column",
    "unsafe_cast_columns",
    "is_numeric",
    "compute_entropy",
    "filter_nulls",
    "set_nulls",
    "set_nulls_series",
    "table_info",
    "and_",
    "or_",
    # auxiliary Functions
    "strip_whitespace_table",
    "strip_whitespace_array",
]

from collections.abc import Sequence

import pandas as pd
import polars as pl
import pyarrow as pa
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
from typing_extensions import Literal, overload

from tsdm.types.dtypes import PYARROW_TO_POLARS


def is_string_array(arr: Array, /) -> bool:
    """Check if an array is a string array."""
    STRING_TYPES = frozenset({pa.string(), pa.large_string()})

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
def arrow_strip_whitespace(obj: Array, /, *cols: str) -> Array: ...  # type: ignore[misc]
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
def arrow_where(mask: BooleanScalar, x: Scalar, y: Scalar = ..., /) -> Scalar: ...
@overload
def arrow_where(  # type: ignore[misc]
    mask: BooleanArray | BooleanScalar,
    x: Array | Scalar,
    y: Array | Scalar = ...,
    /,
) -> Array: ...
def arrow_where(mask, x, y=NA, /):
    """Select elements from x or y depending on mask.

    arrow_where(mask, x, y) is roughly equivalent to x.where(mask, y).
    """
    pa.compute.replace_with_mask(x, mask, y)


@overload
def force_cast(x: Table, dtype: DataType, /) -> Table: ...
@overload
def force_cast(x: Array, dtype: DataType, /) -> Array: ...  # type: ignore[misc]
@overload
def force_cast(x: Table, /, **dtypes: DataType) -> Table: ...
def force_cast(x, dtype=None, /, **dtypes):
    """Cast an array or table to the given data type, replacing non-castable elements with null."""
    x = x.combine_chunks()  # deals with chunked arrays

    if isinstance(x, Array):
        assert dtype is not None and not dtypes
        return (
            pl.from_arrow(x)
            .cast(PYARROW_TO_POLARS[dtype], strict=False)
            .to_arrow()
            .cast(dtype)
        )

    if unknown_keys := set(dtypes.keys()) - set(x.column_names):
        raise ValueError(f"Keys: {unknown_keys} not in table columns.")

    schema: pa.Schema = x.schema
    current_dtypes = dict(zip(schema.names, schema.types, strict=True))
    new_schema = pa.schema(current_dtypes | dtypes)

    return pa.table(
        {
            name: force_cast(x[name], dtypes.get(name) or current_dtypes[name])
            for name in x.column_names
        },
    ).cast(new_schema)


def cast_column(table: Table, col: str, dtype: DataType, /, *, safe: bool) -> Table:
    """Concatenate columns into a new column."""
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
    """Cast columns to the given data types."""
    schema: pa.Schema = table.schema
    current_dtypes = dict(zip(schema.names, schema.types, strict=True))
    if unknown_keys := set(dtypes.keys()) - set(current_dtypes.keys()):
        raise ValueError(f"Keys: {unknown_keys} not in table columns.")

    new_dtypes = current_dtypes | dtypes

    for col, dtype in new_dtypes.items():
        table = cast_column(table, col, dtype, safe=True)
    return table


def unsafe_cast_columns(table: Table, /, **dtypes: DataType) -> Table:
    schema: pa.Schema = table.schema
    current_dtypes = dict(zip(schema.names, schema.types, strict=True))
    if unknown_keys := set(dtypes.keys()) - set(current_dtypes.keys()):
        raise ValueError(f"Keys: {unknown_keys} not in table columns.")

    new_dtypes = current_dtypes | dtypes

    for col, dtype in new_dtypes.items():
        table = cast_column(table, col, dtype, safe=False)
    return table


def is_numeric(array: Array, /) -> Array:
    """Return mask determining if each element can be cast to the given data type."""
    prior_null = pa.compute.is_null(array)
    post_null = pa.compute.is_null(
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

    return pa.compute.or_(
        prior_null,
        pa.compute.invert(post_null),
    )


def compute_entropy(value_counts: Array, /) -> float:
    r"""Compute the normalized entropy using a value_counts array.

    .. math:: ∑_{i=1}^n -pᵢ \log₂(pᵢ)/\log₂(n)

    Note:
        Since entropy is maximized for a uniform distribution, and the entropy
        of a uniform distribution of n choices is log₂(n), the normalization
        ensures that the entropy is in the range [0, 1].
    """
    counts = pa.compute.struct_field(value_counts, 1)

    freqs = pa.compute.divide(
        pa.compute.cast(counts, pa.float64()),
        pa.compute.sum(counts),
    )

    H = pa.compute.divide(
        pa.compute.sum(pa.compute.multiply(freqs, pa.compute.log2(freqs))),
        pa.compute.log2(len(counts)),
    )
    return -H.as_py()


def or_(masks: Sequence[Array], /) -> Array:
    """Compute the logical OR of a sequence of boolean arrays."""
    match n := len(masks):
        case 0:
            return pa.array([])
        case 1:
            return masks[0]
        case _:
            return pa.compute.or_(or_(masks[: n // 2]), or_(masks[n // 2 :]))


def and_(masks: Sequence[Array], /) -> Array:
    """Compute the logical AND of a sequence of boolean arrays."""
    match n := len(masks):
        case 0:
            return pa.array([])
        case 1:
            return masks[0]
        case _:
            return pa.compute.and_(and_(masks[: n // 2]), and_(masks[n // 2 :]))


def filter_nulls(
    table: Table, /, *cols: str, aggregation: Literal["or", "and"] = "or"
) -> Table:
    """Filter rows with null values in the given columns."""
    agg = {"or": or_, "and": and_}[aggregation]
    masks = [table[col].is_null() for col in cols]
    mask = pa.compute.invert(agg(masks))
    return table.filter(mask)


def set_nulls_series(series: Array, values: Sequence, /) -> Array:
    """Set values to null if they match any of the given values."""
    mask = pa.compute.is_in(series, pa.array(values, type=series.type))
    return pa.compute.replace_with_mask(series, mask, pa.null())


def set_nulls(table: Table, /, **cols: Sequence) -> Table:
    """For given columns set all matching values ito null."""
    for col, values in cols.items():
        table = table.set_column(col, set_nulls_series(table[col], values))

    return table


def table_info(table: Table, /) -> None:
    """Print information about a table."""
    size = table.nbytes / (1024 * 1024 * 1024)
    print(f"shape={table.shape}  {size=:.3f} GiB")
    M = max(map(len, table.column_names)) + 1
    for name, col in tqdm(zip(table.column_names, table.columns, strict=True)):
        num_total = len(col)
        num_null = pa.compute.sum(pa.compute.is_null(col)).as_py()
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
            f"{name:{M}s}  {nulls=:s}  {num_uniques=:9d} ({uniques:8.3%})"
            f"  {entropy=:8.3%}  {dtype=:s}"
        )
