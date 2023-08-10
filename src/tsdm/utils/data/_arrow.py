"""Data utilities for Apache Arrow."""

__all__ = [
    "cast_columns",
    "compute_entropy",
    "filter_nulls",
    "table_info",
]

import logging
from collections.abc import Sequence
from typing import Literal

import pandas as pd
import pyarrow as pa
from tqdm.autonotebook import tqdm

__logger__ = logging.getLogger(__name__)


def cast_columns(
    table: pa.Table, force: bool = False, /, **dtypes: pa.DataType
) -> pa.Table:
    """Cast columns to the given data types."""
    schema: pa.Schema = table.schema
    current_dtypes = dict(zip(schema.names, schema.types))
    new_schema = pa.schema(current_dtypes | dtypes)
    return table.cast(new_schema)


def is_numeric(array: pa.Array, /) -> pa.Array:
    """Return mask determining if each element can be cast to the given data type."""
    prior_null = pa.compute.is_null(array)
    post_null = pa.compute.is_null(
        pa.Array.from_pandas(
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


# def force_cast_array(array: pa.Array, /, *, dtype: pa.DataType) -> pa.Array:
#     """Cast an array to the given data type, repalacing non-castable elements with null."""
#     # FIXME: https://github.com/apache/arrow/issues/20486
#     # FIXME: https://github.com/apache/arrow/issues/34976
#     return array.filter(is_numeric(array, dtype=dtype))
#
#
# def force_cast_table(table: pa.Table, /, **dtypes: pa.DataType) -> pa.Table:
#     """Cast a Table to the given data types, repalacing non-castable elements with null."""
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


def compute_entropy(value_counts: pa.Array) -> float:
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


def or_(masks: Sequence[pa.Array]) -> pa.Array:
    """Compute the logical OR of a sequence of boolean arrays."""
    match n := len(masks):
        case 0:
            return pa.array([])
        case 1:
            return masks[0]
        case _:
            return pa.compute.or_(or_(masks[: n // 2]), or_(masks[n // 2 :]))


def and_(masks: Sequence[pa.Array]) -> pa.Array:
    """Compute the logical AND of a sequence of boolean arrays."""
    match n := len(masks):
        case 0:
            return pa.array([])
        case 1:
            return masks[0]
        case _:
            return pa.compute.and_(and_(masks[: n // 2]), and_(masks[n // 2 :]))


def filter_nulls(
    table: pa.Table, cols: list[str], *, aggregation: Literal["or", "and"] = "or"
) -> pa.Table:
    """Filter rows with null values in the given columns."""
    agg = {"or": or_, "and": and_}[aggregation]
    masks = [table[col].is_null() for col in cols]
    mask = pa.compute.invert(agg(masks))
    return table.filter(mask)


def table_info(table: pa.Table) -> None:
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


# def trim_whitespace(table: pa.Table) -> pa.Table:
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
