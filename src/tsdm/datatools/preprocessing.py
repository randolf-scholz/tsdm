r"""Utility functions that act on tabular data."""

__all__ = [
    # types
    "BoundaryInformation",
    # Functions
    "aggregate_nondestructive",
    "select_outliers",
    "is_integer_series",
    "get_integer_cols",
    "remove_outliers",
    "strip_whitespace",
]

import logging
from collections.abc import Mapping
from typing import Any, TypedDict, overload

import pandas as pd
import pyarrow as pa

import tsdm.backend as B
from tsdm.constants import UNDEFINED


class BoundaryInformation(TypedDict):
    r"""Information about the boundaries of a single variable."""

    lower_bound: float | None
    upper_bound: float | None
    lower_inclusive: bool | None
    upper_inclusive: bool | None


@overload
def select_outliers(s: pd.Series, limits: BoundaryInformation, /) -> pd.Series: ...
@overload
def select_outliers(
    s: pd.Series,
    /,
    *,
    lower_bound: float | None,
    upper_bound: float | None,
    lower_inclusive: bool | None,
    upper_inclusive: bool | None,
) -> pd.Series: ...
@overload
def select_outliers(
    df: pd.DataFrame, limits: pd.DataFrame | Mapping[str, BoundaryInformation], /
) -> pd.DataFrame: ...
@overload
def select_outliers[Key](
    df: pd.DataFrame,
    /,
    *,
    lower_bound: Mapping[Key, float | None],
    upper_bound: Mapping[Key, float | None],
    lower_inclusive: Mapping[Key, bool | None],
    upper_inclusive: Mapping[Key, bool | None],
) -> pd.DataFrame: ...
def select_outliers[T: pd.Series | pd.DataFrame](
    obj: T,
    limits: Any = UNDEFINED,
    /,
    *,
    lower_bound: Mapping[Any, float | None] | float | None = UNDEFINED,
    upper_bound: Mapping[Any, float | None] | float | None = UNDEFINED,
    lower_inclusive: Mapping[Any, bool | None] | bool | None = UNDEFINED,
    upper_inclusive: Mapping[Any, bool | None] | bool | None = UNDEFINED,
) -> T:
    r"""Detect outliers in a pd.Series or pd.DataFrame, given boundary values."""
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
        case pd.Series() as s:
            return B.pandas.select_outliers_series(s, **options)
        case pd.DataFrame() as df:
            return B.pandas.select_outliers_dataframe(df, **options)
        case _:
            raise TypeError(f"Unsupported type: {type(obj)}")


@overload
def remove_outliers(
    s: pd.Series,
    limits: BoundaryInformation,
    /,
    *,
    drop: bool = ...,
    inplace: bool = ...,
) -> pd.Series: ...
@overload
def remove_outliers(
    s: pd.Series,
    /,
    *,
    lower_bound: float | None,
    upper_bound: float | None,
    lower_inclusive: bool | None,
    upper_inclusive: bool | None,
    drop: bool = ...,
    inplace: bool = ...,
) -> pd.Series: ...
@overload
def remove_outliers(
    df: pd.DataFrame,
    limits: pd.DataFrame | Mapping[str, BoundaryInformation],
    /,
    *,
    drop: bool = ...,
    inplace: bool = ...,
) -> pd.DataFrame: ...
@overload
def remove_outliers[Key](
    df: pd.DataFrame,
    /,
    *,
    lower_bound: Mapping[Key, float | None],
    upper_bound: Mapping[Key, float | None],
    lower_inclusive: Mapping[Key, bool | None],
    upper_inclusive: Mapping[Key, bool | None],
    drop: bool = ...,
    inplace: bool = ...,
) -> pd.DataFrame: ...
def remove_outliers[T: pd.Series | pd.DataFrame](
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
    r"""Remove outliers from a pd.DataFrame, given boundary values."""
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
        case pd.Series() as s:
            return B.pandas.remove_outliers_series(
                s, drop=drop, inplace=inplace, **options
            )
        case pd.DataFrame() as df:
            return B.pandas.remove_outliers_dataframe(
                df, drop=drop, inplace=inplace, **options
            )
        case _:
            raise TypeError(f"Expected Series or DataFrame, got {type(obj)}")


def strip_whitespace[T: pa.Array | pa.Table | pd.Series | pd.DataFrame](
    table: T, /, *cols: str
) -> T:
    r"""Strip whitespace from all string columns in a table or frame."""
    match table:
        case pa.Table() as table:
            return B.pyarrow.strip_whitespace_table(table, *cols)
        case pa.Array() as array:
            if cols:
                raise ValueError("Cannot specify columns for an Array.")
            return B.pyarrow.strip_whitespace_array(array)
        case pd.Series() as series:
            if cols:
                raise ValueError("Cannot specify columns for a pd.Series.")
            return B.pandas.strip_whitespace_series(series)
        case pd.DataFrame() as frame:
            return B.pandas.strip_whitespace_dataframe(frame, *cols)
        case _:
            raise TypeError(f"Unsupported type: {type(table)}")


def is_integer_series(s: pd.Series, /) -> bool:
    r"""Check if all float values are integral."""
    mask = pd.notna(s)
    return s[mask].apply(float.is_integer).all().item()


def get_integer_cols(table: pd.DataFrame, /) -> set[str]:
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


def aggregate_nondestructive(df: pd.DataFrame, /) -> pd.DataFrame:
    r"""Aggregate multiple simulataneous measurements in a non-destructive way.

    Given a `pd.DataFrame` of size $m×k$, this will construct a new pd.DataFrame of size $m'×k$,
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
    result = pd.DataFrame(index=df.index[:nrows], columns=df.columns)
    result = result.astype(df.dtypes)

    for col in result.columns:
        result[col].iloc[: nitems[col]] = df[col].loc[mask[col]]
    return result
