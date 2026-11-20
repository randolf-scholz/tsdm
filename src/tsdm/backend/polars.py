r"""Polars backend implementation."""

__all__ = [
    "cast",
    "drop_null",
    "is_null",
    "nanmax",
    "nanmin",
    "remove_outliers",
    "scalar",
    "select_outliers",
    "strip_whitespace",
]

from collections.abc import Mapping
from typing import Any, cast as type_cast, overload

import polars as pl
from polars import DataFrame, Series

from tsdm.types.aliases import Axis

from .util import get_outlier_options


def scalar(x: Any, /, dtype: Any) -> Any:
    r"""Return a scalar of a given dtype."""
    return Series([x]).cast(dtype).item()


def nanmin(x: Series, /, axis: Axis = None) -> Any:
    r"""Return the minimum of a Series, ignoring null values.

    Only `axis=None` is supported. Unlike null values, `NaN` values are not
    explicitly filtered by this wrapper.
    """
    if axis is not None:
        raise ValueError("Axis is not supported for polars Series.")
    return x.min()


def nanmax(x: Series, /, axis: Axis = None) -> Any:
    r"""Return the maximum of a Series, ignoring null values.

    Only `axis=None` is supported. Unlike null values, `NaN` values are not
    explicitly filtered by this wrapper.
    """
    if axis is not None:
        raise ValueError("Axis is not supported for polars Series.")
    return x.max()


@overload
def cast(s: Series, /, dtype: Any) -> Series: ...
@overload
def cast(df: DataFrame, /, dtype: Any) -> DataFrame: ...
def cast[T: (Series, DataFrame)](x: T, /, dtype: Any) -> T:
    r"""Cast a polars object to a different dtype."""
    return x.cast(dtype)  # ty: ignore[invalid-explicit-return]


@overload
def drop_null(x: Series, /) -> Series: ...
@overload
def drop_null(x: DataFrame, /) -> DataFrame: ...
def drop_null[T: (Series, DataFrame)](x: T, /) -> T:
    r"""Drop null values, or rows containing nulls for a DataFrame.

    Floating-point `NaN` values are retained.
    """
    return x.drop_nulls()  # ty: ignore[invalid-explicit-return]


def is_null(x: Series, /) -> Series:
    r"""Return a mask identifying null values in a Series.

    Floating-point `NaN` values are not marked as null.
    """
    return x.is_null()


def _select_outliers_series(
    s: Series,
    /,
    *,
    lower_bound: float | None,
    upper_bound: float | None,
    lower_inclusive: bool | None,
    upper_inclusive: bool | None,
) -> Series:
    r"""Return a boolean mask identifying values outside the supplied bounds.

    A bound of `None` leaves that side unbounded. Inclusive bounds admit values
    equal to the boundary. Nulls and values in nonnumeric Series are not marked
    as outliers.
    """
    if not s.dtype.is_numeric():
        return Series(s.name, [False] * len(s), dtype=pl.Boolean)

    match lower_bound, lower_inclusive:
        case None, _:
            mask_lower = Series(s.name, [False] * len(s), dtype=pl.Boolean)
        case _, True:
            mask_lower = (s < lower_bound).fill_null(value=False)
        case _, False:
            mask_lower = (s <= lower_bound).fill_null(value=False)
        case _:
            raise ValueError("Invalid combination of lower_bound and lower_inclusive.")

    match upper_bound, upper_inclusive:
        case None, _:
            mask_upper = Series(s.name, [False] * len(s), dtype=pl.Boolean)
        case _, True:
            mask_upper = (s > upper_bound).fill_null(value=False)
        case _, False:
            mask_upper = (s >= upper_bound).fill_null(value=False)
        case _:
            raise ValueError("Invalid combination of upper_bound and upper_inclusive.")

    return mask_lower | mask_upper


def _select_outliers_dataframe(
    df: DataFrame,
    /,
    *,
    lower_bound: Mapping[str, float | None],
    upper_bound: Mapping[str, float | None],
    lower_inclusive: Mapping[str, bool | None],
    upper_inclusive: Mapping[str, bool | None],
) -> DataFrame:
    r"""Detect outliers in a DataFrame, given boundary values."""
    given_bounds = set.intersection(
        *(
            set(bounds)
            for bounds in (lower_bound, upper_bound, lower_inclusive, upper_inclusive)
        )
    )
    if missing_bounds := set(df.columns) - given_bounds:
        raise ValueError(f"Columns {missing_bounds} do not have bounds!")

    return DataFrame(
        {
            column: _select_outliers_series(
                df.get_column(column),
                lower_bound=lower_bound[column],
                upper_bound=upper_bound[column],
                lower_inclusive=lower_inclusive[column],
                upper_inclusive=upper_inclusive[column],
            )
            for column in df.columns
        }
    )


def _remove_outliers_series[S: Series](
    s: S,
    /,
    *,
    drop: bool = True,
    lower_bound: float | None,
    upper_bound: float | None,
    lower_inclusive: bool | None,
    upper_inclusive: bool | None,
) -> S:
    r"""Remove outliers from a Series, given boundary values.

    Outliers are replaced with nulls and, when `drop=True`, removed. If both
    bounds are `None`, return the input unchanged.

    The input is not modified.
    """
    if lower_bound is None and upper_bound is None:
        return s
    if (
        lower_bound is not None
        and upper_bound is not None
        and lower_bound > upper_bound
    ):
        raise ValueError(
            f"Lower bound {lower_bound} is greater than upper bound {upper_bound}."
        )

    mask = _select_outliers_series(
        s,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lower_inclusive=lower_inclusive,
        upper_inclusive=upper_inclusive,
    )
    result = s.set(mask, None)
    return type_cast("S", result.filter(~mask) if drop else result)


def _remove_outliers_dataframe[T: DataFrame](
    df: T,
    /,
    *,
    drop: bool = True,
    lower_bound: Mapping[str, float | None],
    upper_bound: Mapping[str, float | None],
    lower_inclusive: Mapping[str, bool | None],
    upper_inclusive: Mapping[str, bool | None],
) -> T:
    r"""Remove outliers from a DataFrame, given boundary values.

    Outlying cells are replaced with nulls. When `drop=True`, remove rows only
    if every bounded column is marked as an outlier. Each bounded column must
    occur in all four mappings. Unknown columns raise `ValueError`.

    The input is not modified.
    """
    given_bounds = set.union(
        *(
            set(bounds)
            for bounds in (lower_bound, upper_bound, lower_inclusive, upper_inclusive)
        )
    )
    if extra_bounds := given_bounds - set(df.columns):
        raise ValueError(f"Bounds for {extra_bounds} provided, but no such columns!")
    if not given_bounds:
        return df
    columns = [column for column in df.columns if column in given_bounds]

    mask = _select_outliers_dataframe(
        df.select(columns),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lower_inclusive=lower_inclusive,
        upper_inclusive=upper_inclusive,
    )
    result = df.with_columns(
        pl.when(mask.get_column(column))
        .then(None)
        .otherwise(pl.col(column))
        .alias(column)
        for column in mask.columns
    )
    if not drop:
        return type_cast("T", result)

    row_mask = mask.select(pl.all_horizontal(pl.all()).alias("outlier")).to_series()
    return type_cast("T", result.filter(~row_mask))


@overload
def select_outliers(
    obj: Series, limits: DataFrame | Mapping[str, Any], /
) -> Series: ...
@overload
def select_outliers(
    obj: DataFrame, limits: DataFrame | Mapping[str, Any], /
) -> DataFrame: ...
@overload
def select_outliers(
    obj: Series,
    /,
    *,
    lower_bound: float | None,
    upper_bound: float | None,
    lower_inclusive: bool | None,
    upper_inclusive: bool | None,
) -> Series: ...
@overload
def select_outliers(
    obj: DataFrame,
    /,
    *,
    lower_bound: Mapping[Any, float | None],
    upper_bound: Mapping[Any, float | None],
    lower_inclusive: Mapping[Any, bool | None],
    upper_inclusive: Mapping[Any, bool | None],
) -> DataFrame: ...
def select_outliers(
    obj: Series | DataFrame,
    limits: DataFrame | Mapping[str, Any] | None = None,
    /,
    *,
    lower_bound: Mapping[Any, float | None] | float | None = None,
    upper_bound: Mapping[Any, float | None] | float | None = None,
    lower_inclusive: Mapping[Any, bool | None] | bool | None = None,
    upper_inclusive: Mapping[Any, bool | None] | bool | None = None,
) -> Series | DataFrame:
    r"""Detect outliers using a limits table/mapping or keyword bounds.

    Supply either `limits` or all four keyword bounds. A `None` bound leaves
    that side unbounded. Limits tables must be Polars tables and identify
    columns with `variable`. Mappings may contain either the four bound names
    or per-variable boundary information. With `limits`, all boundary keywords
    must be `None`. Finite bounds require an explicit inclusivity flag.
    """
    if isinstance(limits, DataFrame):
        columns = limits.to_dict(as_series=False)
        variables = columns.pop("variable")
        limits = {
            key: dict(zip(variables, values, strict=True))
            for key, values in columns.items()
        }
    options = get_outlier_options(
        limits,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lower_inclusive=lower_inclusive,
        upper_inclusive=upper_inclusive,
    )
    match obj:
        case Series():
            return _select_outliers_series(obj, **options)
        case DataFrame():
            return _select_outliers_dataframe(obj, **options)
        case _:
            raise TypeError(f"Expected Series or DataFrame, got {type(obj)}.")


@overload
def remove_outliers[A: Series](
    obj: A, limits: DataFrame | Mapping[str, Any], /, *, drop: bool = ...
) -> A: ...
@overload
def remove_outliers[F: DataFrame](
    obj: F, limits: DataFrame | Mapping[str, Any], /, *, drop: bool = ...
) -> F: ...
@overload
def remove_outliers[A: Series](
    obj: A,
    /,
    *,
    lower_bound: float | None,
    upper_bound: float | None,
    lower_inclusive: bool | None,
    upper_inclusive: bool | None,
    drop: bool = ...,
) -> A: ...
@overload
def remove_outliers[F: DataFrame](
    obj: F,
    /,
    *,
    lower_bound: Mapping[Any, float | None],
    upper_bound: Mapping[Any, float | None],
    lower_inclusive: Mapping[Any, bool | None],
    upper_inclusive: Mapping[Any, bool | None],
    drop: bool = ...,
) -> F: ...
def remove_outliers[T: Series | DataFrame](
    obj: T,
    limits: DataFrame | Mapping[str, Any] | None = None,
    /,
    *,
    lower_bound: Mapping[Any, float | None] | float | None = None,
    upper_bound: Mapping[Any, float | None] | float | None = None,
    lower_inclusive: Mapping[Any, bool | None] | bool | None = None,
    upper_inclusive: Mapping[Any, bool | None] | bool | None = None,
    drop: bool = True,
) -> T:
    r"""Remove outliers using a limits table/mapping or keyword bounds.

    Supply either `limits` or all four keyword bounds. A `None` bound leaves
    that side unbounded. Limits tables must be Polars tables and identify
    columns with `variable`. Mappings may contain either the four bound names
    or per-variable boundary information. With `limits`, all boundary keywords
    must be `None`. Finite bounds require an explicit inclusivity flag.

    For tables, supply complete bounds for any subset of columns. Other columns
    are retained unchanged. Bounds for unknown columns raise `ValueError`.
    With `drop=True`, drop rows where every bounded column is an outlier.
    """
    if isinstance(limits, DataFrame):
        columns = limits.to_dict(as_series=False)
        variables = columns.pop("variable")
        limits = {
            key: dict(zip(variables, values, strict=True))
            for key, values in columns.items()
        }
    options = get_outlier_options(
        limits,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lower_inclusive=lower_inclusive,
        upper_inclusive=upper_inclusive,
    )
    match obj:
        case Series():
            return _remove_outliers_series(obj, drop=drop, **options)
        case DataFrame():
            return _remove_outliers_dataframe(obj, drop=drop, **options)
        case _:
            raise TypeError(f"Expected Series or DataFrame, got {type(obj)}.")


@overload
def strip_whitespace[S: Series](obj: S, /) -> S: ...
@overload
def strip_whitespace[F: DataFrame](obj: F, /, *cols: str) -> F: ...
def strip_whitespace(obj: Series | DataFrame, /, *cols: str) -> Series | DataFrame:
    r"""Trim string values, optionally restricting processing to selected columns.

    Nonstrings and null values are preserved. Column names are only accepted
    for DataFrames.
    """
    match obj:
        case DataFrame():
            columns = cols or obj.columns
            return obj.with_columns(
                pl.col(column).str.strip_chars()
                for column in columns
                if obj.schema[column] == pl.String
            )
        case Series():
            if cols:
                raise ValueError("Cannot specify columns for a Series.")
            return obj.str.strip_chars() if obj.dtype == pl.String else obj
        case _:
            raise TypeError(f"Expected Series or DataFrame, got {type(obj)}.")
