r"""Utility functions that act on tabular data."""

__all__ = [
    # types
    "BoundaryInformation",
    "InlineTable",
    # Functions
    "aggregate_nondestructive",
    "select_outliers",
    "is_integer_series",
    "get_integer_cols",
    "make_dataframe",
    "remove_outliers",
    "strip_whitespace",
]

import logging
from collections.abc import Mapping, Sequence
from typing import Any, NotRequired, Optional, Required, TypedDict, overload

import pandas as pd
import pyarrow as pa
from pandas import DataFrame, Series
from scipy import stats

from tsdm.backend.pandas import (
    remove_outliers_dataframe,
    remove_outliers_series,
    select_outliers_dataframe,
    select_outliers_series,
    strip_whitespace_dataframe,
    strip_whitespace_series,
)
from tsdm.backend.pyarrow import strip_whitespace_array, strip_whitespace_table
from tsdm.constants import UNDEFINED


class BoundaryInformation(TypedDict):
    r"""Information about the boundaries of a single variable."""

    lower_bound: float | None
    upper_bound: float | None
    lower_inclusive: bool | None
    upper_inclusive: bool | None


class TableMetadata(BoundaryInformation):
    r"""Table schema."""

    name: str
    dtype: str
    unit: str | None
    lower_bound: float | None
    upper_bound: float | None
    lower_inclusive: bool | None
    upper_inclusive: bool | None
    description: str | None


class InlineTable[*Ts](TypedDict):
    r"""A table of data in a dictionary."""

    data: Required[Sequence[tuple[*Ts]]]
    schema: Mapping[str, Any]
    index: NotRequired[str | list[str]]


def make_dataframe(
    data: Sequence[tuple[Any, ...]],
    *,
    columns: Optional[list[str]] = None,
    dtypes: Optional[list[str | type]] = None,
    schema: Optional[Mapping[str, Any]] = None,
    index: Optional[str | list[str]] = None,
) -> DataFrame:
    r"""Make a DataFrame from a dictionary."""
    if dtypes is not None and schema is not None:
        raise ValueError("Cannot specify both dtypes and schema.")

    if columns is not None:
        cols = list(columns)
    elif schema is not None:
        cols = list(schema)
    else:
        cols = None

    df = DataFrame.from_records(data, columns=cols)

    if dtypes is not None:
        df = df.astype(dict(zip(df.columns, dtypes, strict=True)))
    elif schema is not None:
        df = df.astype(schema)

    if index is not None:
        df = df.set_index(index)

    return df


def strip_whitespace[T: pa.Array | pa.Table | Series | DataFrame](
    table: T, /, *cols: str
) -> T:
    r"""Strip whitespace from all string columns in a table or frame."""
    match table:
        case pa.Table() as table:
            return strip_whitespace_table(table, *cols)
        case pa.Array() as array:
            if cols:
                raise ValueError("Cannot specify columns for an Array.")
            return strip_whitespace_array(array)
        case Series() as series:
            if cols:
                raise ValueError("Cannot specify columns for a Series.")
            return strip_whitespace_series(series)  # pyright: ignore[reportReturnType]
        case DataFrame() as frame:
            return strip_whitespace_dataframe(frame, *cols)  # pyright: ignore[reportReturnType]
        case _:
            raise TypeError(f"Unsupported type: {type(table)}")


# region overloads ---------------------------------------------------------------------
@overload
def select_outliers(s: Series, limits: BoundaryInformation, /) -> Series: ...
@overload
def select_outliers(
    s: Series,
    /,
    *,
    lower_bound: float | None,
    upper_bound: float | None,
    lower_inclusive: bool | None,
    upper_inclusive: bool | None,
) -> Series: ...
@overload
def select_outliers(
    df: DataFrame, limits: DataFrame | Mapping[str, BoundaryInformation], /
) -> DataFrame: ...
@overload
def select_outliers[Key](
    df: DataFrame,
    /,
    *,
    lower_bound: Mapping[Key, float | None],
    upper_bound: Mapping[Key, float | None],
    lower_inclusive: Mapping[Key, bool | None],
    upper_inclusive: Mapping[Key, bool | None],
) -> DataFrame: ...
# endregion overloads ------------------------------------------------------------------
def select_outliers[T: Series | DataFrame](
    obj: T,
    limits: Any = UNDEFINED,
    /,
    *,
    lower_bound: Mapping[Any, float | None] | float | None = UNDEFINED,
    upper_bound: Mapping[Any, float | None] | float | None = UNDEFINED,
    lower_inclusive: Mapping[Any, bool | None] | bool | None = UNDEFINED,
    upper_inclusive: Mapping[Any, bool | None] | bool | None = UNDEFINED,
) -> T:
    r"""Detect outliers in a Series or DataFrame, given boundary values."""
    lims: Mapping[str, Any] = {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "lower_inclusive": lower_inclusive,
        "upper_inclusive": upper_inclusive,
    }
    undefined = [val is UNDEFINED for val in lims.values()]

    if limits is UNDEFINED:
        if any(undefined):
            raise ValueError(f"Missing boundary values: {lims}")
        options = lims
    elif all(undefined):
        options = {key: limits[key] for key in lims}
    else:
        raise ValueError("Limits specified both as positional and keyword arguments.")

    match obj:
        case Series() as s:
            return select_outliers_series(s, **options)
        case DataFrame() as df:
            return select_outliers_dataframe(df, **options)
        case _:
            raise TypeError(f"Unsupported type: {type(obj)}")


# region overloads ---------------------------------------------------------------------
@overload
def remove_outliers(
    s: Series,
    limits: BoundaryInformation,
    /,
    *,
    drop: bool = ...,
    inplace: bool = ...,
) -> Series: ...
@overload
def remove_outliers(
    s: Series,
    /,
    *,
    lower_bound: float | None,
    upper_bound: float | None,
    lower_inclusive: bool | None,
    upper_inclusive: bool | None,
    drop: bool = ...,
    inplace: bool = ...,
) -> Series: ...
@overload
def remove_outliers(
    df: DataFrame,
    limits: DataFrame | Mapping[str, BoundaryInformation],
    /,
    *,
    drop: bool = ...,
    inplace: bool = ...,
) -> DataFrame: ...
@overload
def remove_outliers[Key](
    df: DataFrame,
    /,
    *,
    lower_bound: Mapping[Key, float | None],
    upper_bound: Mapping[Key, float | None],
    lower_inclusive: Mapping[Key, bool | None],
    upper_inclusive: Mapping[Key, bool | None],
    drop: bool = ...,
    inplace: bool = ...,
) -> DataFrame: ...
# endregion overloads ------------------------------------------------------------------
def remove_outliers[T: Series | DataFrame](
    obj: T,
    limits: Any = UNDEFINED,
    /,
    *,
    lower_bound: Mapping[Any, float | None] | float | None = UNDEFINED,
    upper_bound: Mapping[Any, float | None] | float | None = UNDEFINED,
    lower_inclusive: Mapping[Any, bool | None] | bool | None = UNDEFINED,
    upper_inclusive: Mapping[Any, bool | None] | bool | None = UNDEFINED,
    drop: bool = True,
    inplace: bool = False,
) -> T:
    r"""Remove outliers from a DataFrame, given boundary values."""
    lims: Mapping[str, Any] = {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "lower_inclusive": lower_inclusive,
        "upper_inclusive": upper_inclusive,
    }
    undefined = [val is UNDEFINED for val in lims.values()]

    if limits is UNDEFINED:
        if any(undefined):
            raise ValueError(f"Missing boundary values: {lims}")
        options = lims
    elif all(undefined):
        options = {key: limits[key] for key in lims}
    else:
        raise ValueError("Limits specified both as positional and keyword arguments.")

    match obj:
        case Series() as s:
            return remove_outliers_series(s, drop=drop, inplace=inplace, **options)
        case DataFrame() as df:
            return remove_outliers_dataframe(df, drop=drop, inplace=inplace, **options)
        case _:
            raise TypeError(f"Expected Series or DataFrame, got {type(obj)}")


def is_integer_series(s: Series, /) -> bool:
    r"""Check if all float values are integral."""
    mask = pd.notna(s)
    return s[mask].apply(float.is_integer).all().item()


def get_integer_cols(table: DataFrame, /) -> set[str]:
    r"""Get all columns that contain only integers."""
    logger = logging.getLogger(f"{__name__}/get_integer_cols")

    cols: set[str] = set()
    for col in table.columns:
        if pd.api.types.is_integer_dtype(table[col]):
            logger.debug("Integer column                       : %s", col)
            cols.add(col)
        elif pd.api.types.is_float_dtype(table[col]) and is_integer_series(table[col]):
            logger.debug("Integer column pretending to be float: %s", col)
            cols.add(col)
    return cols


def aggregate_nondestructive(df: DataFrame, /) -> DataFrame:
    r"""Aggregate multiple simulataneous measurements in a non-destructive way.

    Given a `DataFrame` of size $m×k$, this will construct a new DataFrame of size $m'×k$,
    where `m' = max(df.notna().sum())` is the maximal number of measured not-null values.

    For example::

                              Acetate  Base   DOT  Fluo_GFP   Glucose  OD600  Probe_Volume    pH
        elapsed_time
        2020-12-09 09:48:38      <NA>  <NA>  <NA>      <NA>  4.578233   <NA>          <NA>  <NA>
        2020-12-09 09:48:38      <NA>  <NA>  <NA>      <NA>      <NA>  0.445          <NA>  <NA>
        2020-12-09 09:48:38  0.116585  <NA>  <NA>      <NA>      <NA>   <NA>          <NA>  <NA>
        2020-12-09 09:48:38  0.114842  <NA>  <NA>      <NA>      <NA>   <NA>          <NA>  <NA>
        2020-12-09 09:48:38      <NA>  <NA>  <NA>      <NA>      <NA>  0.485          <NA>  <NA>
        2020-12-09 09:48:38      <NA>  <NA>  <NA>      <NA>      <NA>   <NA>           200  <NA>
        2020-12-09 09:48:38      <NA>  <NA>  <NA>    1112.5      <NA>   <NA>          <NA>  <NA>
        2020-12-09 09:48:38      <NA>  <NA>  <NA>     912.5      <NA>   <NA>          <NA>  <NA>
        2020-12-09 09:48:38      <NA>  <NA>  <NA>      <NA>  4.554859   <NA>          <NA>  <NA>

    becomes::

                              Acetate  Base   DOT  Fluo_GFP   Glucose  OD600  Probe_Volume    pH
        elapsed_time
        2020-12-09 09:48:38  0.116585  <NA>  <NA>    1112.5  4.578233  0.445           200  <NA>
        2020-12-09 09:48:38  0.114842  <NA>  <NA>     912.5  4.554859  0.485          <NA>  <NA>
    """
    mask = df.notna()
    nitems = mask.sum()
    nrows = nitems.max()
    result = DataFrame(index=df.index[:nrows], columns=df.columns)
    result = result.astype(df.dtypes)

    for col in result.columns:
        result[col].iloc[: nitems[col]] = df[col].loc[mask[col]]
    return result


def describe(
    s: Series | DataFrame,
    /,
    *,
    quantiles: tuple[float, ...] = (0, 0.01, 0.5, 0.99, 1),
) -> DataFrame:
    r"""Describe a DataFrame on a per column basis."""
    if isinstance(s, DataFrame):
        df = s
        return pd.concat([describe(df[col]) for col in df])

    # crete a zero-scalar of the same type as the series
    num = len(s)

    # compute the special values
    max_value = s.max()
    min_value = s.min()
    mode_value = s.mode().iloc[0]

    try:
        mean_value = s.mean()
        median_value = s.median()
        std_value = s.std()
    except Exception:
        mean_value = float("nan")
        std_value = float("nan")
        median_value = float("nan")

    # compute counts
    max_count = (s == max_value).sum()
    min_count = (s == min_value).sum()
    mode_count = (s == mode_value).sum()
    nan_count = s.isna().sum()
    unique_count = s.nunique()

    try:
        idx = s.first_valid_index()
        # NOTE: dropna is necessary for duplicate index
        val = s.loc[idx].dropna().iloc[0] if idx is not None else 0
        zero = val - val
        neg_count = (s < zero).sum()
        pos_count = (s > zero).sum()
        zero_count = (s == zero).sum()
    except Exception:
        neg_count = pd.NA
        pos_count = pd.NA
        zero_count = pd.NA

    # compute the rates
    max_rate = max_count / num
    min_rate = min_count / num
    mode_rate = mode_count / num
    nan_rate = nan_count / num
    neg_rate = neg_count / num
    pos_rate = pos_count / num
    unique_rate = unique_count / num
    zero_rate = zero_count / num

    # stats
    try:
        entropy = stats.entropy(s.value_counts(), base=2)
    except Exception:
        entropy = pd.NA
    try:
        quantile_values = s.quantile(quantiles)
    except Exception:
        quantile_values = Series([float("nan")] * len(quantiles))

    return DataFrame(
        {
            # stats
            ("stats", "entropy"): entropy,
            **{
                ("quantile", f"{q:.2f}"): v
                for q, v in zip(quantiles, quantile_values, strict=True)
            },
            # special values
            ("value", "max"): max_value,
            ("value", "mean"): mean_value,
            ("value", "median"): median_value,
            ("value", "min"): min_value,
            ("value", "mode"): mode_value,
            ("value", "std"): std_value,
            # counts
            ("count", "max"): max_count,
            ("count", "min"): min_count,
            ("count", "mode"): mode_count,
            ("count", "nan"): nan_count,
            ("count", "neg"): neg_count,
            ("count", "pos"): pos_count,
            ("count", "unique"): unique_count,
            ("count", "zero"): zero_count,
            # rates
            ("rate", "max"): max_rate,
            ("rate", "min"): min_rate,
            ("rate", "mode"): mode_rate,
            ("rate", "nan"): nan_rate,
            ("rate", "neg"): neg_rate,
            ("rate", "pos"): pos_rate,
            ("rate", "unique"): unique_rate,
            ("rate", "zero"): zero_rate,
        },
        index=[s.name],
    )


def aggregate_set[T](data: tuple[T, ...], /) -> T:
    r"""Coerces multiple values if they are identical."""
    try:
        vals = set(data)
    except TypeError as exc:
        exc.add_note("Data not hashable, please provide an aggregate_fn.")
        raise

    if len(vals) != 1:
        raise ValueError("Data not constant, please provide an aggregate_fn.")
    return vals.pop()
