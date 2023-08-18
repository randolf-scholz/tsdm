"""implement pandas backend for tsdm."""

__all__ = [
    # Functions
    "pandas_infer_axes",
    "pandas_clip",
    "pandas_false_like",
    "pandas_like",
    "pandas_nanmax",
    "pandas_nanmean",
    "pandas_nanmin",
    "pandas_nanstd",
    "pandas_strip_whitespace",
    "pandas_true_like",
    "pandas_null_like",
    "pandas_where",
    # auxiliary functions
    "strip_whitespace_series",
    "strip_whitespace_dataframe",
]

from typing import Literal, TypeVar

from numpy.typing import ArrayLike, NDArray
from pandas import NA, DataFrame, Series

from tsdm.types.aliases import Axes
from tsdm.types.variables import scalar_co as Scalar_co

P = TypeVar("P", Series, DataFrame)
"""A type variable for pandas objects."""


def pandas_false_like(x: P, /) -> P:
    """Returns a constant boolean tensor with the same shape/device as `x`."""
    m = x.isna()
    return m ^ m


def pandas_true_like(x: P, /) -> P:
    """Returns a constant boolean tensor with the same shape/device as `x`."""
    m = x.isna()
    return m ^ (~m)


def pandas_infer_axes(
    x: P, /, *, axes: Axes = None
) -> Literal[None, "index", "columns"]:
    """Convert axes specification to pandas-compatible axes specification.

    - Series: -1 → 0, -2 → Error
    - DataFrame: -1 → 1, -2 → 0
    """
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
    return "columns" if axes % len(x.shape) else "index"


def pandas_clip(x: P, lower: NDArray | None, upper: NDArray | None, /) -> P:
    """Analogue to `numpy.clip`."""
    axis = "columns" if isinstance(x, DataFrame) else "index"
    return x.clip(lower, upper, axis=axis)


def pandas_nanmax(x: P, /, *, axis: Axes = None) -> P:
    """Analogue to `numpy.nanmax`."""
    return x.max(axis=pandas_infer_axes(x, axes=axis), skipna=True)


def pandas_nanmin(x: P, /, *, axis: Axes = None) -> P:
    """Analogue to `numpy.nanmin`."""
    return x.min(axis=pandas_infer_axes(x, axes=axis), skipna=True)


def pandas_nanmean(x: P, /, *, axis: Axes = None) -> P:
    """Analogue to `numpy.nanmean`."""
    return x.mean(axis=pandas_infer_axes(x, axes=axis), skipna=True)


def pandas_nanstd(x: P, /, *, axis: Axes = None) -> P:
    """Analogue to `numpy.nanstd`."""
    return x.std(axis=pandas_infer_axes(x, axes=axis), skipna=True, ddof=0)


def pandas_where(cond: NDArray, a: P, b: Scalar_co | NDArray, /) -> P:
    """Analogue to `numpy.where`."""
    if isinstance(a, Series | DataFrame):
        return a.where(cond, b)
    return a if cond else pandas_like(b, a)  # scalar fallback


def pandas_null_like(x: P, /) -> P:
    """Returns a copy of the input filled with nulls."""
    return pandas_where(pandas_true_like(x), x, NA)


def pandas_like(x: ArrayLike, ref: P, /) -> P:
    """Create a Series/DataFrame with the same modality as a reference."""
    if isinstance(ref, Series):
        return Series(x, dtype=ref.dtype, index=ref.index)
    if isinstance(ref, DataFrame):
        return DataFrame(x, index=ref.index, columns=ref.columns).astype(ref.dtypes)
    raise TypeError(f"Expected Series or DataFrame, got {type(ref)}.")


def strip_whitespace_dataframe(frame: DataFrame, /, *cols: str) -> DataFrame:
    """Strip whitespace from selected columns in a DataFrame."""
    return frame.assign(
        **{col: strip_whitespace_series(frame[col]) for col in (cols or frame)}
    )


def strip_whitespace_series(series: Series, /) -> Series:
    """Strip whitespace from all string elements in a Series."""
    if series.dtype == "string":
        return series.str.strip()
    return series


def pandas_strip_whitespace(x: P, /) -> P:
    """Strip whitespace from all string elements in a pandas object."""
    if isinstance(x, DataFrame):
        return strip_whitespace_dataframe(x)
    if isinstance(x, Series):
        return strip_whitespace_series(x)
    raise TypeError(f"Expected Series or DataFrame, got {type(x)}.")
