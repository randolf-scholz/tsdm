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
    "is_string_array",
    "null_like",
    "or_",
    "remove_outliers_table",
    "remove_outliers_array",
    "scalar",
    "select_outliers_table",
    "select_outliers_array",
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

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, Optional, cast, overload

import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import (
    NA,
    Array,
    BooleanArray,
    BooleanScalar,
    ChunkedArray,
    DataType,
    DictionaryArray,
    ListArray,
    Scalar,
    Table,
)
from pyarrow.types import lib as pyarrow_lib
from tqdm import tqdm

STR = pa.string()
TEXT = pa.large_string()
STRING_TYPES = frozenset({STR, TEXT})

type AnyArray = Array | ChunkedArray
type BooleanAnyArray = BooleanArray | ChunkedArray
type Mask = bool | list[bool] | BooleanArray | BooleanScalar


def scalar(x: Any, /, dtype: DataType | str) -> Scalar:
    return pa.scalar(x, type=dtype)


def strip_whitespace_table[T: Table](table: T, /, *cols: str) -> T:
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


def strip_whitespace_array[A: AnyArray](arr: A, /) -> A:
    r"""Strip whitespace from all string elements in an array."""
    match arr:
        case ChunkedArray(chunks=chunks):
            return pa.chunked_array(map(strip_whitespace_array, chunks))
        case ListArray(type=dtype) if dtype.value_type in STRING_TYPES:
            return pa.array(map(pc.utf8_trim_whitespace, arr), type=dtype)
        case DictionaryArray(type=dtype, indices=indices, dictionary=dictionary) if (
            dtype.value_type in STRING_TYPES
        ):
            return DictionaryArray.from_arrays(
                indices,
                pc.utf8_trim_whitespace(dictionary),
            )
        case Array(type=dtype) if dtype in STRING_TYPES:
            return pc.utf8_trim_whitespace(arr)
        case _:
            raise TypeError(f"Expected string array, got {arr.type}.")


@overload
def strip_whitespace[A: AnyArray](obj: A, /) -> A: ...
@overload
def strip_whitespace[T: Table](obj: T, /, *cols: str) -> T: ...
def strip_whitespace[T: Table | AnyArray](obj: T, /, *cols: str) -> T:
    r"""Strip whitespace from all string elements in an arrow object."""
    match obj:
        case Table() as table:
            return strip_whitespace_table(table, *cols)
        case (Array() | ChunkedArray()) as array:
            if cols:
                raise ValueError("Cannot specify columns for an Array.")
            return strip_whitespace_array(array)
        case _:
            raise TypeError(f"Expected Array or Table, got {type(obj)}.")


def false_like(arr: AnyArray, /) -> BooleanArray:
    r"""Creates a `BooleanArray` of False values with the same length as arr."""
    m = arr.is_valid()
    return pc.xor(m, m)


def true_like(arr: AnyArray, /) -> BooleanArray:
    r"""Creates a `BooleanArray` of True values with the same length as arr."""
    return pc.invert(false_like(arr))


def full_like[A: AnyArray](arr: A, /, *, fill_value: Scalar) -> A:
    r"""Creates an `Array` of `fill_value` with the same length as arr."""
    if not isinstance(fill_value, Scalar):
        fill_value = pa.scalar(fill_value)
    if fill_value is NA:
        fill_value = fill_value.cast(arr.type)
    if fill_value.type == arr.type:
        return pc.replace_with_mask(arr, false_like(arr), fill_value)
    empty = null_like(arr).cast(fill_value.type)
    return pc.replace_with_mask(empty, true_like(arr), fill_value)


def null_like[A: AnyArray](arr: A, /) -> A:
    r"""Creates an `Array` of null-values with the same length as arr."""
    return full_like(arr, fill_value=NA)


def where[T: AnyArray](mask: Mask, x: T | Scalar, y: T | Scalar = NA, /) -> T:
    r"""Select elements from x or y depending on mask.

    arrow_where(mask, x, y) is roughly equivalent to x.where(mask, y).
    """
    return pc.replace_with_mask(x, mask, y)


def _is_real_numeric_array(arr: AnyArray, /) -> bool:
    r"""Check whether an Arrow array contains real numeric values."""
    return (
        pa.types.is_integer(arr.type)
        or pa.types.is_floating(arr.type)
        or pa.types.is_decimal(arr.type)
    )


def select_outliers_array(
    s: AnyArray,
    /,
    *,
    lower_bound: float | None,
    upper_bound: float | None,
    lower_inclusive: bool | None,
    upper_inclusive: bool | None,
) -> BooleanAnyArray:
    r"""Detect outliers in an Arrow array, given boundary values."""
    if not _is_real_numeric_array(s):
        return false_like(s)

    values = pc.cast(s, pa.float64())
    false = pa.scalar(value=False)
    match lower_bound, lower_inclusive:
        case None, _:
            mask_lower = false_like(s)
        case _, True:
            mask_lower = pc.fill_null(pc.less(values, lower_bound), false)
        case _, False:
            mask_lower = pc.fill_null(pc.less_equal(values, lower_bound), false)
        case _:
            raise ValueError("Invalid combination of lower_bound and lower_inclusive.")

    match upper_bound, upper_inclusive:
        case None, _:
            mask_upper = false_like(s)
        case _, True:
            mask_upper = pc.fill_null(pc.greater(values, upper_bound), false)
        case _, False:
            mask_upper = pc.fill_null(pc.greater_equal(values, upper_bound), false)
        case _:
            raise ValueError("Invalid combination of upper_bound and upper_inclusive.")

    return pc.or_(mask_lower, mask_upper)


def select_outliers_table(
    df: Table,
    /,
    *,
    lower_bound: Mapping[str, float | None],
    upper_bound: Mapping[str, float | None],
    lower_inclusive: Mapping[str, bool | None],
    upper_inclusive: Mapping[str, bool | None],
) -> Table:
    r"""Detect outliers in an Arrow table, given boundary values."""
    given_bounds = set.intersection(
        *(
            set(bounds)
            for bounds in (lower_bound, upper_bound, lower_inclusive, upper_inclusive)
        )
    )
    if missing_bounds := set(df.column_names) - given_bounds:
        raise ValueError(f"Columns {missing_bounds} do not have bounds!")

    return pa.table(
        {
            column: select_outliers_array(
                df[column],
                lower_bound=lower_bound[column],
                upper_bound=upper_bound[column],
                lower_inclusive=lower_inclusive[column],
                upper_inclusive=upper_inclusive[column],
            )
            for column in df.column_names
        }
    )


def remove_outliers_array[A: AnyArray](
    s: A,
    /,
    *,
    drop: bool = True,
    inplace: bool = False,
    lower_bound: float | None,
    upper_bound: float | None,
    lower_inclusive: bool | None,
    upper_inclusive: bool | None,
) -> A:
    r"""Remove outliers from an Arrow array, given boundary values.

    ``inplace`` is accepted for API compatibility. Arrow arrays are immutable.
    """
    del inplace
    if lower_bound is None and upper_bound is None:
        return s
    if isinstance(s, ChunkedArray):
        result = remove_outliers_array(
            s.combine_chunks(),
            drop=drop,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            lower_inclusive=lower_inclusive,
            upper_inclusive=upper_inclusive,
        )
        return cast("A", pa.chunked_array([result]))
    if (
        lower_bound is not None
        and upper_bound is not None
        and lower_bound > upper_bound
    ):
        raise ValueError(
            f"Lower bound {lower_bound} is greater than upper bound {upper_bound}."
        )

    mask = select_outliers_array(
        s,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lower_inclusive=lower_inclusive,
        upper_inclusive=upper_inclusive,
    )
    result = pc.replace_with_mask(s, mask, pa.scalar(None, type=s.type))
    return pc.filter(result, pc.invert(mask)) if drop else result


def remove_outliers_table(
    df: Table,
    /,
    *,
    drop: bool = True,
    inplace: bool = False,
    lower_bound: Mapping[str, float | None],
    upper_bound: Mapping[str, float | None],
    lower_inclusive: Mapping[str, bool | None],
    upper_inclusive: Mapping[str, bool | None],
    erroron_extra_bounds: bool = False,
) -> Table:
    r"""Remove outliers from an Arrow table, given boundary values.

    ``inplace`` is accepted for API compatibility. Arrow tables are immutable.
    """
    del inplace
    given_bounds = set.intersection(
        *(
            set(bounds)
            for bounds in (lower_bound, upper_bound, lower_inclusive, upper_inclusive)
        )
    )
    if missing_bounds := set(df.column_names) - given_bounds:
        raise ValueError(f"Columns {missing_bounds} do not have bounds!")
    if erroron_extra_bounds and (extra_bounds := given_bounds - set(df.column_names)):
        raise ValueError(f"Bounds for {extra_bounds} provided, but no such columns!")

    mask = select_outliers_table(
        df,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lower_inclusive=lower_inclusive,
        upper_inclusive=upper_inclusive,
    )
    result = df
    for column in df.column_names:
        result = result.set_column(
            result.column_names.index(column),
            column,
            pc.replace_with_mask(
                df[column].combine_chunks(),
                mask[column].combine_chunks(),
                pa.scalar(None, type=df[column].type),
            ),
        )

    if not drop:
        return result

    row_mask = and_(mask[column].combine_chunks() for column in mask.column_names)
    return result.filter(pc.invert(row_mask))


@overload
def force_cast[A: AnyArray | Table](x: A, dtype: DataType | str, /) -> A: ...
@overload
def force_cast(x: Table, /, **dtypes: DataType | str) -> Table: ...
def force_cast[T: AnyArray | Table](
    x: T, dtype: Optional[DataType | str] = None, /, **dtypes: DataType | str
) -> T:
    r"""Cast an array or table to the given data type, replacing non-castable elements with null."""
    match x:
        case (Array() | ChunkedArray()) as array:
            if dtypes:
                raise ValueError("Unexpected argument dtypes for Array input.")
            if dtype is None:
                raise ValueError("Must specify dtype for Array input.")

            actual_dtype = pyarrow_lib.ensure_type(dtype)

            return array.cast(
                options=pc.CastOptions(
                    target_type=actual_dtype,
                    allow_float_truncate=True,
                    allow_decimal_truncate=True,
                    allow_time_truncate=True,
                    allow_invalid_utf8=True,
                ),
            )

        case Table() as table:
            if unknown_keys := set(dtypes.keys()) - set(table.column_names):
                raise ValueError(f"Keys: {unknown_keys} not in table columns.")

            schema: pa.Schema = table.schema
            current_dtypes = dict(zip(schema.names, schema.types, strict=True))
            new_schema = pa.schema(current_dtypes | dtypes)

            return pa.table(
                {
                    name: force_cast(
                        table[name], dtypes.get(name) or current_dtypes[name]
                    )
                    for name in table.column_names
                }
            ).cast(new_schema)

        case _:
            raise TypeError(f"Expected Array or Table, got {type(x)}.")


def cast_column(
    table: Table, col: str, dtype: DataType | str, /, *, safe: bool
) -> Table:
    r"""Concatenate columns into a new column."""
    try:
        casted_column = (
            table[col].combine_chunks().dictionary_encode()
            if isinstance(dtype, pa.DictionaryType)
            else table[col].cast(dtype, safe=safe)
        )
    except Exception as exc:
        exc.add_note(
            f"Error {exc!r} occurred while casting column {col!r} to {dtype!r}."
        )
        raise

    index = table.column_names.index(col)
    return table.set_column(index, col, casted_column)


def cast_columns(table: Table, /, **dtypes: DataType | str) -> Table:
    r"""Cast columns to the given data types."""
    schema: pa.Schema = table.schema
    current_dtypes = dict(zip(schema.names, schema.types, strict=True))
    if unknown_keys := set(dtypes.keys()) - set(current_dtypes.keys()):
        raise ValueError(f"Keys: {unknown_keys} not in table columns.")

    new_dtypes = current_dtypes | dtypes

    for col, dtype in new_dtypes.items():
        table = cast_column(table, col, dtype, safe=True)
    return table


def unsafe_cast_columns(table: Table, /, **dtypes: DataType | str) -> Table:
    r"""Cast columns to the given data types, replacing non-castable elements with null."""
    schema: pa.Schema = table.schema
    current_dtypes = dict(zip(schema.names, schema.types, strict=True))
    if unknown_keys := set(dtypes.keys()) - set(current_dtypes.keys()):
        raise ValueError(f"Keys: {unknown_keys} not in table columns.")

    new_dtypes = current_dtypes | dtypes

    for col, dtype in new_dtypes.items():
        table = cast_column(table, col, dtype, safe=False)
    return table


def is_string_array(arr: AnyArray, /) -> bool:
    r"""Check if an array is a string array."""
    match arr.type:
        case _ if arr.type in STRING_TYPES:
            return True
        case pa.ListType(value_type=value_type):
            return value_type in STRING_TYPES
        case pa.DictionaryType(value_type=value_type):
            return value_type in STRING_TYPES
        case _:
            return False


def compute_entropy(value_counts: AnyArray, /) -> float:
    r"""Compute the normalized entropy using a value_counts array.

    .. math:: ∑ᵢ₌₁ⁿ -pᵢ \log₂(pᵢ)/\log₂(n)

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


def or_(masks: Iterable[BooleanArray], /) -> BooleanArray:
    r"""Compute the logical OR of a sequence of boolean arrays."""
    iterator = iter(masks)
    try:
        result = next(iterator)
    except StopIteration:
        return pa.array([], type=pa.bool_())
    for mask in iterator:
        result = pc.or_(result, mask)
    return result


def and_(masks: Iterable[BooleanArray], /) -> BooleanArray:
    r"""Compute the logical AND of a sequence of boolean arrays."""
    iterator = iter(masks)
    try:
        result = next(iterator)
    except StopIteration:
        return pa.array([], type=pa.bool_())
    for mask in iterator:
        result = pc.and_(result, mask)
    return result


def filter_nulls(
    table: Table, /, *cols: str, aggregation: Literal["or", "and"] = "or"
) -> Table:
    r"""Filter rows with null values in the given columns."""
    agg = {"or": or_, "and": and_}[aggregation]
    mask = pc.invert(agg(table[col].is_null() for col in cols))
    return table.filter(mask)


def set_nulls_series[A: AnyArray](series: A, values: Sequence, /) -> A:
    r"""Set values to null if they match any of the given values."""
    if isinstance(series, ChunkedArray):
        return pa.chunked_array(set_nulls_series(arr, values) for arr in series.chunks)

    if isinstance(series, DictionaryArray):
        ref_type = series.type.value_type
        mask = pc.is_in(series, pa.array(values, type=ref_type))
        null = pa.scalar(None, type=ref_type)
        result = pc.replace_with_mask(series.dictionary_decode(), mask, null)
        return result.dictionary_encode()

    mask = pc.is_in(series, pa.array(values, type=series.type))
    null = pa.scalar(None, type=series.type)
    return pc.replace_with_mask(series, mask, null)


def set_nulls(table: Table, /, **cols: Sequence) -> Table:
    r"""For given columns set all matching values ito null."""
    for col, values in cols.items():
        table = table.set_column(
            table.column_names.index(col),
            col,
            set_nulls_series(table[col], values),
        )

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
