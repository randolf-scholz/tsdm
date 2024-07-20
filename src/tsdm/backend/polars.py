r"""Implement `polars`-backend for tsdm."""

__all__ = [
    "nanmin",
    "nanmax",
    "cast",
    "drop_null",
    "is_null",
    "scalar",
]

from typing import Any, overload

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
def cast(x, /, dtype):
    r"""Cast a polars object to a different dtype."""
    return x.cast(dtype)


@overload
def drop_null(x: Series, /) -> Series: ...
@overload
def drop_null(x: DataFrame, /) -> DataFrame: ...
def drop_null(x, /):
    r"""Drop `NaN` values from a polars object."""
    return x.drop_nulls()


def is_null(x: Series, /) -> Series:
    r"""Check for `NaN` values in a polars object."""
    return x.is_null()
