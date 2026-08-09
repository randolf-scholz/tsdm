r"""Implement `polars`-backend for tsdm."""

__all__ = [
    "nanmin",
    "nanmax",
    "cast",
    "drop_null",
    "is_null",
    "remove_outliers_dataframe",
    "remove_outliers_series",
    "scalar",
    "select_outliers_dataframe",
    "select_outliers_series",
]

from collections.abc import Mapping
from typing import Any, cast as type_cast, overload

import polars as pl
from polars import DataFrame, Series

from tsdm.types.aliases import Axis


def scalar(x: Any, /, dtype: Any) -> Any:
    r"""Return a scalar of a given dtype."""
    return Series([x]).cast(dtype).item()


def nanmin(x: Series, /, axis: Axis = None) -> Any:
    r"""Analogue to `numpy.nanmin`."""
    if axis is not None:
        raise ValueError("Axis is not supported for polars Series.")
    return x.min()


def nanmax(x: Series, /, axis: Axis = None) -> Any:
    r"""Analogue to `numpy.nanmin`."""
    if axis is not None:
        raise ValueError("Axis is not supported for polars Series.")
    return x.max()


@overload
def cast(s: Series, /, dtype: Any) -> Series: ...
@overload
def cast(df: DataFrame, /, dtype: Any) -> DataFrame: ...
def cast[T: (Series, DataFrame)](x: T, /, dtype: Any) -> T:
    r"""Cast a polars object to a different dtype."""
    return x.cast(dtype)


@overload
def drop_null(x: Series, /) -> Series: ...
@overload
def drop_null(x: DataFrame, /) -> DataFrame: ...
def drop_null[T: (Series, DataFrame)](x: T, /) -> T:
    r"""Drop `NaN` values from a polars object."""
    return x.drop_nulls()


def is_null(x: Series, /) -> Series:
    r"""Check for `NaN` values in a polars object."""
    return x.is_null()


def select_outliers_series(
    s: Series,
    /,
    *,
    lower_bound: float | None,
    upper_bound: float | None,
    lower_inclusive: bool | None,
    upper_inclusive: bool | None,
) -> Series:
    r"""Detect outliers in a Series, given boundary values."""
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


def select_outliers_dataframe(
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
            column: select_outliers_series(
                df.get_column(column),
                lower_bound=lower_bound[column],
                upper_bound=upper_bound[column],
                lower_inclusive=lower_inclusive[column],
                upper_inclusive=upper_inclusive[column],
            )
            for column in df.columns
        }
    )


def remove_outliers_series[S: Series](
    s: S,
    /,
    *,
    drop: bool = True,
    inplace: bool = False,
    lower_bound: float | None,
    upper_bound: float | None,
    lower_inclusive: bool | None,
    upper_inclusive: bool | None,
) -> S:
    r"""Remove outliers from a Series, given boundary values.

    ``inplace`` is accepted for API compatibility. Polars objects are immutable.
    """
    del inplace
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

    mask = select_outliers_series(
        s,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lower_inclusive=lower_inclusive,
        upper_inclusive=upper_inclusive,
    )
    result = s.set(mask, None)
    return type_cast("S", result.filter(~mask) if drop else result)


def remove_outliers_dataframe[T: DataFrame](
    df: T,
    /,
    *,
    drop: bool = True,
    inplace: bool = False,
    lower_bound: Mapping[str, float | None],
    upper_bound: Mapping[str, float | None],
    lower_inclusive: Mapping[str, bool | None],
    upper_inclusive: Mapping[str, bool | None],
    erroron_extra_bounds: bool = False,
) -> T:
    r"""Remove outliers from a DataFrame, given boundary values.

    ``inplace`` is accepted for API compatibility. Polars objects are immutable.
    """
    del inplace
    given_bounds = set.intersection(
        *(
            set(bounds)
            for bounds in (lower_bound, upper_bound, lower_inclusive, upper_inclusive)
        )
    )
    if missing_bounds := set(df.columns) - given_bounds:
        raise ValueError(f"Columns {missing_bounds} do not have bounds!")
    if erroron_extra_bounds and (extra_bounds := given_bounds - set(df.columns)):
        raise ValueError(f"Bounds for {extra_bounds} provided, but no such columns!")

    mask = select_outliers_dataframe(
        df,
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
        for column in df.columns
    )
    if not drop:
        return type_cast("T", result)

    row_mask = mask.select(pl.all_horizontal(pl.all()).alias("outlier")).to_series()
    return type_cast("T", result.filter(~row_mask))
