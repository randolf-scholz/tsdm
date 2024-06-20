r"""Implement `polars`-backend for tsdm."""

__all__ = [
    "nanmin",
    "nanmax",
    "cast",
    "drop_null",
    "is_null",
    "scalar",
]

from polars import DataFrame, Series
from typing_extensions import Any, TypeVar

from tsdm.types.aliases import Axis

PL = TypeVar("PL", DataFrame, Series)


def scalar(x: Any, /, dtype: Any) -> Any:
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


def cast(x: PL, /, dtype: Any) -> PL:
    r"""Cast a polars object to a different dtype."""
    return x.cast(dtype)


def drop_null(x: PL, /) -> PL:
    r"""Drop `NaN` values from a polars object."""
    return x.drop_nulls()


def is_null(x: Series, /) -> Series:
    r"""Check for `NaN` values in a polars object."""
    return x.is_null()
