"""implement pandas backend for tsdm."""

__all__ = [
    "pandas_axes",
    "pandas_clip",
    "pandas_nanmax",
    "pandas_nanmean",
    "pandas_nanmin",
    "pandas_nanstd",
    "pandas_where",
]

from typing import Literal, TypeVar

from numpy.typing import NDArray
from pandas import DataFrame, Series

from tsdm.types.aliases import Axes

P = TypeVar("P", Series, DataFrame)
"""A type variable for pandas objects."""


def pandas_axes(
    shape: tuple[int, ...], axes: Axes = None
) -> Literal[None, "index", "columns"]:
    """Get the axis of a pandas object."""
    match axes:
        case None:
            return None
        case int():
            pass
        case tuple():
            if len(axes) != 1:
                raise ValueError(f"Expected 1 axis, got {len(axes)}.")
            axes = axes[0]
        case _:
            raise TypeError(f"Expected int or Iterable[int], got {type(axes)}.")
    return "columns" if axes % len(shape) else "index"


def pandas_clip(x: P, lower: None | NDArray = None, upper: None | NDArray = None) -> P:
    """Analogue to `numpy.clip`."""
    axis = "columns" if isinstance(x, DataFrame) else "index"
    return x.clip(lower, upper, axis=axis)


def pandas_nanmax(x: P, axis: Axes = None) -> P:
    """Analogue to `numpy.nanmax`."""
    return x.max(axis=pandas_axes(x.shape, axis), skipna=True)


def pandas_nanmin(x: P, axis: Axes = None) -> P:
    """Analogue to `numpy.nanmin`."""
    return x.min(axis=pandas_axes(x.shape, axis), skipna=True)


def pandas_nanmean(x: P, axis: Axes = None) -> P:
    """Analogue to `numpy.nanmean`."""
    return x.mean(axis=pandas_axes(x.shape, axis), skipna=True)


def pandas_nanstd(x: P, axis: Axes = None) -> P:
    """Analogue to `numpy.nanstd`."""
    return x.std(axis=pandas_axes(x.shape, axis), skipna=True, ddof=0)


def pandas_where(cond: NDArray, a: P, b: NDArray) -> NDArray:
    """Analogue to `numpy.where`."""
    if isinstance(a, Series | DataFrame):
        return a.where(cond, b)
    return a if cond else b  # scalar fallback
