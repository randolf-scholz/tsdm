"""Implement `pandas`-backend for tsdm."""

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
    "strip_whitespace_index",
    "strip_whitespace_series",
    "strip_whitespace_dataframe",
]

from numpy.typing import ArrayLike, NDArray
from pandas import NA, DataFrame, Index, Series
from typing_extensions import Literal, TypeAlias, TypeVar

from tsdm.types.aliases import Axes, Scalar

P = TypeVar("P", Index, Series, DataFrame)
"""A type variable for pandas objects."""

PANDAS_TYPE: TypeAlias = Index | Series | DataFrame
"""A type alias for pandas objects."""


def pandas_false_like(x: P, /) -> P:
    """Returns a constant boolean tensor with the same shape/device as `x`."""
    m = x.isna()
    return m ^ m


def pandas_true_like(x: P, /) -> P:
    """Returns a constant boolean tensor with the same shape/device as `x`."""
    m = x.isna()
    return m ^ (~m)


def pandas_infer_axes(
    x: P, /, *, axis: Axes = None
) -> Literal[None, "index", "columns"]:
    """Convert axes specification to pandas-compatible axes specification.

    - Series: -1 → 0, -2 → Error
    - DataFrame: -1 → 1, -2 → 0
    """
    match axis:
        case None:
            return None
        case int():
            pass
        case [int(ax)]:
            axis = ax
        case _:
            raise TypeError(f"Expected int or tuple[int], got {type(axis)}.")
    return "columns" if axis % len(x.shape) else "index"


def pandas_clip(x: P, lower: NDArray | None, upper: NDArray | None, /) -> P:
    """Analogue to `numpy.clip`."""
    axis = "columns" if isinstance(x, DataFrame) else "index"
    return x.clip(lower, upper, axis=axis)


def pandas_nanmax(x: P, /, *, axis: Axes = None) -> P:
    """Analogue to `numpy.nanmax`."""
    return x.max(axis=pandas_infer_axes(x, axis=axis), skipna=True)


def pandas_nanmin(x: P, /, *, axis: Axes = None) -> P:
    """Analogue to `numpy.nanmin`."""
    return x.min(axis=pandas_infer_axes(x, axis=axis), skipna=True)


def pandas_nanmean(x: P, /, *, axis: Axes = None) -> P:
    """Analogue to `numpy.nanmean`."""
    return x.mean(axis=pandas_infer_axes(x, axis=axis), skipna=True)


def pandas_nanstd(x: P, /, *, axis: Axes = None) -> P:
    """Analogue to `numpy.nanstd`."""
    return x.std(axis=pandas_infer_axes(x, axis=axis), skipna=True, ddof=0)


def pandas_where(cond: NDArray, a: P, b: Scalar | NDArray, /) -> P:
    """Analogue to `numpy.where`."""
    if isinstance(a, PANDAS_TYPE):  # type: ignore[misc,arg-type]
        return a.where(cond, b)
    return a if cond else pandas_like(b, a)  # scalar fallback


def pandas_null_like(x: P, /) -> P:
    """Returns a copy of the input filled with nulls."""
    return pandas_where(pandas_true_like(x), x, NA)


def pandas_like(x: ArrayLike, ref: P, /) -> P:
    """Create a Series/DataFrame with the same modality as a reference."""
    match ref:
        case Index() as idx:
            return Index(x, dtype=idx.dtype, name=idx.name)
        case Series() as s:
            return Series(x, dtype=s.dtype, index=s.index)
        case DataFrame() as df:
            return DataFrame(x, index=df.index, columns=df.columns).astype(df.dtypes)
        case _:
            raise TypeError(f"Expected {PANDAS_TYPE}, got {type(ref)}.")


def strip_whitespace_index(index: Index, /) -> Index:
    """Strip whitespace from all string elements in an Index."""
    if index.dtype == "string":
        return index.str.strip()
    return index


def strip_whitespace_series(series: Series, /) -> Series:
    """Strip whitespace from all string elements in a Series."""
    if series.dtype == "string":
        return series.str.strip()
    return series


def strip_whitespace_dataframe(frame: DataFrame, /, *cols: str) -> DataFrame:
    """Strip whitespace from selected columns in a DataFrame."""
    return frame.assign(
        **{col: strip_whitespace_series(frame[col]) for col in (cols or frame)}
    )


def pandas_strip_whitespace(x: P, /) -> P:
    """Strip whitespace from all string elements in a `pandas` object."""
    match x:
        case DataFrame() as df:
            return strip_whitespace_dataframe(df)
        case Series() as s:
            return strip_whitespace_series(s)
        case Index() as idx:
            return strip_whitespace_index(idx)
        case _:
            raise TypeError(f"Expected {PANDAS_TYPE}, got {type(x)}.")
