"""Data utilities for Apache Arrow."""

__all__ = [
    "cast_columns",
    "compute_entropy",
    "filter_nulls",
    "force_cast",
    "table_info",
]

import logging
from collections.abc import Sequence
from typing import Literal, Optional, TypeVar, overload

import pandas as pd
import polars as pl
import pyarrow as pa
from pyarrow import Array, DataType, Table
from tqdm.autonotebook import tqdm

from tsdm.types.dtypes import PYARROW_TO_POLARS

__logger__ = logging.getLogger(__name__)


P = TypeVar("P", Array, Table)


@overload
def force_cast(x: P, dtype: DataType, /) -> P: ...
@overload
def force_cast(x: Table, /, **dtypes: DataType) -> Table: ...
def force_cast(x: P, dtype: Optional[DataType] = None, /, **dtypes: DataType) -> P:
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
    current_dtypes = dict(zip(schema.names, schema.types))
    new_schema = pa.schema(current_dtypes | dtypes)

    return pa.table(
        {
            name: force_cast(x[name], dtypes.get(name) or current_dtypes[name])
            for name in x.column_names
        },
    ).cast(new_schema)


def cast_columns(table: Table, safe: bool = True, /, **dtypes: DataType) -> Table:
    """Cast columns to the given data types."""
    schema: pa.Schema = table.schema
    current_dtypes = dict(zip(schema.names, schema.types))
    if unknown_keys := set(dtypes.keys()) - set(current_dtypes.keys()):
        raise ValueError(f"Keys: {unknown_keys} not in table columns.")

    new_dtypes = current_dtypes | dtypes

    for col, dtype in new_dtypes.items():
        if isinstance(dtype, pa.DictionaryType):
            arr = table[col].combine_chunks()
            table = table.set_column(
                table.column_names.index(col),
                col,
                arr.dictionary_encode(),
            )
        else:
            table = table.set_column(
                table.column_names.index(col),
                col,
                table[col].cast(dtype, safe=safe),
            )
    return table


def is_numeric(array: Array, /) -> Array:
    """Return mask determining if each element can be cast to the given data type."""
    prior_null = pa.compute.is_null(array)
    post_null = pa.compute.is_null(
        Array.from_pandas(
            pd.to_numeric(
                pd.Series(array, dtype="string[pyarrow]"),
                # NOTE: string[pyarrow] actually StringDtype, needed so we don't use to_pandas()
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


# def force_cast_array(array: Array, /, *, dtype: DataType) -> Array:
#     """Cast an array to the given data type, replacing non-castable elements with null."""
#     # FIXME: https://github.com/apache/arrow/issues/20486
#     # FIXME: https://github.com/apache/arrow/issues/34976
#     return array.filter(is_numeric(array, dtype=dtype))
#
#
# def force_cast_table(table: Table, /, **dtypes: DataType) -> Table:
#     """Cast a Table to the given data types, replacing non-castable elements with null."""
#     for index, column in enumerate(table.column_names):
#         if column in dtypes:
#             array = table[column]
#             mask = is_numeric(array)
#             dropped = 1 - pa.compute.mean(mask).as_py()
#             __logger__.info(
#                 "Masking %8.3%% of non-castable values in %s", dropped, column
#             )
#             values = array.filter(mask)
#             table = table.set_column(index, column, values)
#     return table


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


def table_info(table: Table, /) -> None:
    """Print information about a table."""
    size = table.nbytes / (1024 * 1024 * 1024)
    print(f"shape={table.shape}  {size=:.3f} GiB")
    M = max(map(len, table.column_names)) + 1
    for name, col in tqdm(zip(table.column_names, table.columns)):
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


# def trim_whitespace(table: Table) -> Table:
#     for column in table.column_names:
#         array = table[column]
#         index = table.column_names.index(column)
#         dtype = array.type
#         if dtype == (DICT_TYPE, STRING_TYPE, TEXT_TYPE):
#             values = pa.compute.cast(
#                 pa.compute.utf8_trim_whitespace(array.cast("string")),
#                 dtype,
#             )
#             table.set_column(index, column, values)
#     return table
