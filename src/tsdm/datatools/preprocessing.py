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
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict, overload

import pandas as pd
import polars as pl
import pyarrow as pa

import tsdm.backend as B
from tsdm.constants import UNDEFINED


class BoundaryInformation(TypedDict):
    r"""Information about the boundaries of a single variable."""

    lower_bound: float | None
    upper_bound: float | None
    lower_inclusive: bool | None
    upper_inclusive: bool | None


def _limits_from_columns(
    columns: Mapping[str, Sequence[Any]], keys: Sequence[str], /
) -> dict[str, dict[Any, Any]]:
    r"""Create per-variable limit mappings from a metadata table."""
    variables = columns["variable"]
    return {key: dict(zip(variables, columns[key], strict=True)) for key in keys}


# pyrefly: ignore[bad-return]
def _extract_limits(limits: Any, keys: Sequence[str], /) -> Mapping[str, Any]:
    r"""Extract outlier-limit mappings from a backend-specific table."""
    match limits:
        case pl.DataFrame() as frame:
            return _limits_from_columns(frame.to_dict(as_series=False), keys)
        case pa.Table() as table:
            return _limits_from_columns(table.to_pydict(), keys)
        case pd.DataFrame() as frame:
            match "variable" in frame:
                case True:
                    return _limits_from_columns(
                        {column: frame[column].to_list() for column in frame}, keys
                    )
                case False:
                    return {key: frame[key] for key in keys}
        case Mapping() as mapping:
            match set(keys).issubset(mapping):
                case True:
                    return {key: mapping[key] for key in keys}
                case False:
                    return {
                        key: {
                            variable: bounds[key]
                            for variable, bounds in mapping.items()
                        }
                        for key in keys
                    }
        case _:
            return {key: limits[key] for key in keys}


def _get_outlier_options(
    limits: Any,
    /,
    *,
    lower_bound: Mapping[Any, float | None] | float | None,
    upper_bound: Mapping[Any, float | None] | float | None,
    lower_inclusive: Mapping[Any, bool | None] | bool | None,
    upper_inclusive: Mapping[Any, bool | None] | bool | None,
) -> Mapping[str, Any]:
    r"""Validate and normalize outlier bounds for all supported backends."""
    lims: Mapping[str, Any] = {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "lower_inclusive": lower_inclusive,
        "upper_inclusive": upper_inclusive,
    }
    undefined = [value is UNDEFINED for value in lims.values()]

    if limits is UNDEFINED:
        if any(undefined):
            raise ValueError(f"Missing boundary values: {lims}")
        return lims
    if all(undefined):
        return _extract_limits(limits, list(lims))
    raise ValueError("Limits specified both as positional and keyword arguments.")


# TODO: return type not entirely honest (generic type changes from e.g. float to bool)
@overload
def select_outliers(s: pl.Series, limits: BoundaryInformation, /) -> pl.Series: ...
@overload
def select_outliers(s: pa.Array, limits: BoundaryInformation, /) -> pa.Array: ...
@overload
def select_outliers(s: pd.Series, limits: BoundaryInformation, /) -> pd.Series: ...  # pyright: ignore[reportOverlappingOverload]
@overload
def select_outliers(
    s: pl.Series,
    /,
    *,
    lower_bound: float | None,
    upper_bound: float | None,
    lower_inclusive: bool | None,
    upper_inclusive: bool | None,
) -> pl.Series: ...
@overload
def select_outliers(
    s: pa.Array,
    /,
    *,
    lower_bound: float | None,
    upper_bound: float | None,
    lower_inclusive: bool | None,
    upper_inclusive: bool | None,
) -> pa.Array: ...
@overload
def select_outliers(  # pyright: ignore[reportOverlappingOverload]
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
    df: pl.DataFrame, limits: pl.DataFrame | Mapping[str, BoundaryInformation], /
) -> pl.DataFrame: ...
@overload
def select_outliers(
    df: pa.Table, limits: pa.Table | Mapping[str, BoundaryInformation], /
) -> pa.Table: ...
@overload
def select_outliers(  # pyright: ignore[reportOverlappingOverload]
    df: pd.DataFrame, limits: pd.DataFrame | Mapping[str, BoundaryInformation], /
) -> pd.DataFrame: ...
@overload
def select_outliers(
    df: pl.DataFrame,
    /,
    *,
    lower_bound: Mapping[Any, float | None],
    upper_bound: Mapping[Any, float | None],
    lower_inclusive: Mapping[Any, bool | None],
    upper_inclusive: Mapping[Any, bool | None],
) -> pl.DataFrame: ...
@overload
def select_outliers(
    df: pa.Table,
    /,
    *,
    lower_bound: Mapping[Any, float | None],
    upper_bound: Mapping[Any, float | None],
    lower_inclusive: Mapping[Any, bool | None],
    upper_inclusive: Mapping[Any, bool | None],
) -> pa.Table: ...
@overload
def select_outliers(  # pyright: ignore[reportOverlappingOverload]
    df: pd.DataFrame,
    /,
    *,
    lower_bound: Mapping[Any, float | None],
    upper_bound: Mapping[Any, float | None],
    lower_inclusive: Mapping[Any, bool | None],
    upper_inclusive: Mapping[Any, bool | None],
) -> pd.DataFrame: ...
def select_outliers(
    obj: pa.Array | pa.Table | pd.Series | pd.DataFrame | pl.Series | pl.DataFrame,
    limits: Any = UNDEFINED,
    /,
    *,
    lower_bound: Mapping[Any, float | None] | float | None = UNDEFINED,
    upper_bound: Mapping[Any, float | None] | float | None = UNDEFINED,
    lower_inclusive: Mapping[Any, bool | None] | bool | None = UNDEFINED,
    upper_inclusive: Mapping[Any, bool | None] | bool | None = UNDEFINED,
) -> pa.Array | pa.Table | pd.Series | pd.DataFrame | pl.Series | pl.DataFrame:
    r"""Detect outliers in a supported series or table, given boundary values."""
    options = _get_outlier_options(
        limits,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lower_inclusive=lower_inclusive,
        upper_inclusive=upper_inclusive,
    )

    match obj:
        case pl.Series() as s:
            return B.polars.select_outliers_series(s, **options)
        case pl.DataFrame() as df:
            return B.polars.select_outliers_dataframe(df, **options)
        case pa.Array() as array:
            return B.pyarrow.select_outliers_array(array, **options)
        case pa.Table() as table:
            return B.pyarrow.select_outliers_table(table, **options)
        case pd.Series() as s:
            return B.pandas.select_outliers_series(s, **options)
        case pd.DataFrame() as df:
            return B.pandas.select_outliers_dataframe(df, **options)
        case _:
            raise TypeError(f"Expected a supported series or table, got {type(obj)}")


@overload
def remove_outliers[T: pa.Array | pd.Series | pl.Series](
    s: T,
    limits: BoundaryInformation,
    /,
    *,
    drop: bool = ...,
    inplace: bool = ...,
) -> T: ...
@overload
def remove_outliers[T: pa.Array | pd.Series | pl.Series](
    s: T,
    /,
    *,
    lower_bound: float | None,
    upper_bound: float | None,
    lower_inclusive: bool | None,
    upper_inclusive: bool | None,
    drop: bool = ...,
    inplace: bool = ...,
) -> T: ...
@overload
def remove_outliers[T: pa.Table | pd.DataFrame | pl.DataFrame](
    df: T,
    limits: pa.Table | pd.DataFrame | pl.DataFrame | Mapping[str, BoundaryInformation],
    /,
    *,
    drop: bool = ...,
    inplace: bool = ...,
) -> T: ...
@overload
def remove_outliers[T: pa.Table | pd.DataFrame | pl.DataFrame](
    df: T,
    /,
    *,
    lower_bound: Mapping[Any, float | None],
    upper_bound: Mapping[Any, float | None],
    lower_inclusive: Mapping[Any, bool | None],
    upper_inclusive: Mapping[Any, bool | None],
    drop: bool = ...,
    inplace: bool = ...,
) -> T: ...
def remove_outliers[
    T: pa.Array | pa.Table | pd.Series | pd.DataFrame | pl.Series | pl.DataFrame
](
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
    r"""Remove outliers from a supported series or table, given boundary values."""
    options = _get_outlier_options(
        limits,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lower_inclusive=lower_inclusive,
        upper_inclusive=upper_inclusive,
    )

    match obj:
        case pl.Series() as s:
            return B.polars.remove_outliers_series(  # pyright: ignore[reportReturnType]
                s, drop=drop, inplace=inplace, **options
            )
        case pl.DataFrame() as df:
            return B.polars.remove_outliers_dataframe(  # pyright: ignore[reportReturnType]  # pyrefly: ignore[bad-return]
                df, drop=drop, inplace=inplace, **options
            )
        case pa.Array() as array:
            return B.pyarrow.remove_outliers_array(
                array, drop=drop, inplace=inplace, **options
            )
        case pa.Table() as table:
            return B.pyarrow.remove_outliers_table(
                table, drop=drop, inplace=inplace, **options
            )
        case pd.Series() as s:
            return B.pandas.remove_outliers_series(
                s, drop=drop, inplace=inplace, **options
            )
        case pd.DataFrame() as df:
            return B.pandas.remove_outliers_dataframe(
                df, drop=drop, inplace=inplace, **options
            )

        case _:
            raise TypeError(f"Expected a supported series or table, got {type(obj)}")


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
