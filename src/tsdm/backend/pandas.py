r"""Implement `pandas`-backend for tsdm."""

__all__ = [
    # types
    "PandasDTypeArg",
    "PandasDtype",
    "PandasType",
    # Constants
    "NA_VALUES",
    # Functions
    "cast",
    "clip",
    "copy_like",
    "drop_null",
    "false_like",
    "infer_axes",
    "nanmax",
    "nanmean",
    "nanmin",
    "nanstd",
    "null_like",
    "scalar",
    "strip_whitespace",
    "true_like",
    "where",
    # auxiliary functions
    "select_outliers_series",
    "select_outliers_dataframe",
    "remove_outliers_series",
    "remove_outliers_dataframe",
    "strip_whitespace_index",
    "strip_whitespace_series",
    "strip_whitespace_dataframe",
]

import logging
import operator
import warnings
from collections.abc import Mapping
from contextlib import suppress
from functools import reduce
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from pandas import NA, DataFrame, Index, MultiIndex, NaT, Series
from pandas.core.dtypes.base import ExtensionDtype

from tsdm.types.aliases import Axis

__logger__ = logging.getLogger(__name__)

PandasDtype = ExtensionDtype | np.dtype
r"""Type Alias for `pandas` dtype."""
PandasDTypeArg = str | type | PandasDtype
r"""Type Alias for `pandas` dtype arguments."""
type PandasType = DataFrame | Series | Index | MultiIndex
r"""Type Alias for `pandas` objects."""
type PANDAS_TYPE = Index | Series | DataFrame
r"""A type alias for pandas objects."""

NA_VALUES: Final[frozenset[object]] = frozenset({NA, NaT})
r"""Values that correspond to NaN."""


def scalar(x: Any, /, dtype: Any) -> Any:
    r"""Cast a scalar to a different dtype."""
    return Index([x]).astype(dtype).item()


def cast[P: PandasType](x: P, /, dtype: Any) -> P:
    r"""Cast a `pandas` object to a different dtype."""
    return x.astype(dtype)


def drop_null[P: PandasType](x: P, /) -> P:
    r"""Drop `NaN` values from a pandas object."""
    return x.dropna()


def false_like[P: PandasType](x: P, /) -> P:
    r"""Returns a constant boolean tensor with the same shape/device as `x`."""
    m = x.isna()
    return m ^ m


def true_like[P: PandasType](x: P, /) -> P:
    r"""Returns a constant boolean tensor with the same shape/device as `x`."""
    m = x.isna()
    return m ^ (~m)


def infer_axes(
    x: PandasType, /, *, axis: Axis = None
) -> Literal["index", "columns"] | None:
    r"""Convert axes specification to pandas-compatible axes specification.

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


def clip[P: PandasType](x: P, lower: NDArray | None, upper: NDArray | None, /) -> P:
    r"""Analogue to `numpy.clip`."""
    axis = "columns" if isinstance(x, DataFrame) else "index"
    # FIXME: https://github.com/pandas-dev/pandas/issues/59053
    with suppress(Exception):
        lower = lower.item()  # type: ignore
    with suppress(Exception):
        upper = upper.item()  # type: ignore

    return x.clip(lower, upper, axis=axis)


def nanmax[P: PandasType](x: P, /, *, axis: Axis = None) -> P:
    r"""Analogue to `numpy.nanmax`."""
    return x.max(axis=infer_axes(x, axis=axis), skipna=True)


def nanmin[P: PandasType](x: P, /, *, axis: Axis = None) -> P:
    r"""Analogue to `numpy.nanmin`."""
    return x.min(axis=infer_axes(x, axis=axis), skipna=True)


def nanmean[P: PandasType](x: P, /, *, axis: Axis = None) -> P:
    r"""Analogue to `numpy.nanmean`."""
    return x.mean(axis=infer_axes(x, axis=axis), skipna=True)


def nanstd[P: PandasType](x: P, /, *, axis: Axis = None) -> P:
    r"""Analogue to `numpy.nanstd`."""
    return x.std(axis=infer_axes(x, axis=axis), skipna=True, ddof=0)


def where[
    T: (Index, MultiIndex, Series, DataFrame),
](cond: T | Any, a: T | Any, b: T | Any, /) -> T:
    r"""Analogue to `numpy.where`."""
    if isinstance(a, DataFrame):
        return a.where(cond, b, axis=0)
    if isinstance(b, DataFrame):
        return b.where(~cond, a, axis=0)
    if isinstance(cond, DataFrame):
        return cond.where(cond, a, axis=0).where(~cond, b, axis=0)
    if isinstance(a, Series):
        return a.where(cond, b)
    if isinstance(b, Series):
        return b.where(~cond, a)
    if isinstance(cond, Series):
        return cond.where(cond, a).where(~cond, b)
    if isinstance(a, Index):
        return a.where(cond, b)
    if isinstance(b, Index):
        return b.where(~cond, a)
    if isinstance(cond, Index):
        return cond.where(cond, a).where(~cond, b)
    return np.where(cond, a, b)  # pyright: ignore[reportReturnType]


def null_like[P: PandasType](x: P, /) -> P:
    r"""Returns a copy of the input filled with nulls."""
    return where(true_like(x), x, NA)


def copy_like[P: PandasType](x: ArrayLike, ref: P, /) -> P:
    r"""Create a Series/DataFrame with the same modality as a reference."""
    match ref:
        case Index() as idx:
            return Index(x, dtype=idx.dtype, name=idx.name)
        case Series() as s:
            return Series(x, dtype=s.dtype, index=s.index)
        case DataFrame() as df:
            return DataFrame(x, index=df.index, columns=df.columns).astype(df.dtypes)
        case _:
            raise TypeError(f"Expected {PANDAS_TYPE}, got {type(ref)}.")


# region auxiliary functions -----------------------------------------------------------
def strip_whitespace_index[I: Index](index: I, /) -> I:
    r"""Strip whitespace from all string elements in an Index."""
    if index.dtype == "string":
        return index.str.strip()
    return index


def strip_whitespace_series[S: Series](series: S, /) -> S:
    r"""Strip whitespace from all string elements in a Series."""
    if series.dtype == "string":
        return series.str.strip()
    return series


def strip_whitespace_dataframe[DF: DataFrame](frame: DF, /, *cols: str) -> DF:
    r"""Strip whitespace from selected columns in a DataFrame."""
    return frame.assign(
        **{col: strip_whitespace_series(frame[col]) for col in (cols or frame)}
    )


def strip_whitespace[P: PandasType](x: P, /) -> P:
    r"""Strip whitespace from all string elements in a `pandas` object."""
    match x:
        case DataFrame() as df:
            return strip_whitespace_dataframe(df)
        case Series() as s:
            return strip_whitespace_series(s)
        case Index() as idx:
            return strip_whitespace_index(idx)
        case _:
            raise TypeError(f"Expected {PANDAS_TYPE}, got {type(x)}.")


def select_outliers_series(
    s: Series,
    /,
    *,
    lower_bound: float | None,
    upper_bound: float | None,
    lower_inclusive: bool | None,
    upper_inclusive: bool | None,
) -> Series:  # Series[bool]
    r"""Detect outliers in a Series, given boundary values."""
    if not pd.api.types.is_any_real_numeric_dtype(s):
        warnings.warn(
            "Currently, only float-compatible dtypes are support outlier selection.",
            UserWarning,
            stacklevel=2,
        )
        return false_like(s)

    # detect lower-bound violations
    match lower_bound, lower_inclusive:
        case None, _:
            mask_lower = false_like(s)
        case _, True:
            mask_lower = (s < lower_bound).fillna(value=False)
        case _, False:
            mask_lower = (s <= lower_bound).fillna(value=False)
        case _:
            raise ValueError("Invalid combination of lower_bound and lower_inclusive.")

    # detect upper-bound violations
    match upper_bound, upper_inclusive:
        case None, _:
            mask_upper = false_like(s)
        case _, True:
            mask_upper = (s > upper_bound).fillna(value=False)
        case _, False:
            mask_upper = (s >= upper_bound).fillna(value=False)
        case _:
            raise ValueError("Invalid combination of upper_bound and upper_inclusive.")

    if __logger__.getEffectiveLevel() >= logging.INFO:
        lower_rate = f"{mask_lower.mean():8.3%}" if mask_lower.any() else " ------%"
        upper_rate = f"{mask_upper.mean():8.3%}" if mask_upper.any() else " ------%"
        __logger__.info(
            "%s/%s lower/upper bound violations in %r", lower_rate, upper_rate, s.name
        )

    return mask_lower | mask_upper


def select_outliers_dataframe(
    df: DataFrame,
    /,
    *,
    lower_bound: Mapping[Any, float | None],
    upper_bound: Mapping[Any, float | None],
    lower_inclusive: Mapping[Any, bool | None],
    upper_inclusive: Mapping[Any, bool | None],
) -> DataFrame:  # DataFrame[bool]
    r"""Detect outliers in a DataFrame, given boundary values."""
    given_bounds = set.intersection(
        *(
            set(d.keys())
            for d in [lower_bound, upper_bound, lower_inclusive, upper_inclusive]
        )
    )
    if missing_bounds := set(df.columns) - given_bounds:
        raise ValueError(f"Columns {missing_bounds} do not have bounds!")

    mask = false_like(df)
    exceptions: dict[str, Exception] = {}
    for col in df.columns:
        try:
            mask[col] = select_outliers_series(
                df[col],
                lower_bound=lower_bound[col],
                upper_bound=upper_bound[col],
                lower_inclusive=lower_inclusive[col],
                upper_inclusive=upper_inclusive[col],
            )
        except Exception as e:
            exceptions[col] = e
    if exceptions:
        msg = "\n".join(f"{col!r}: {exc!s}" for col, exc in exceptions.items())
        raise RuntimeError(msg) from ExceptionGroup("", list(exceptions.values()))

    return mask


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
    r"""Remove outliers from a Series, given boundary values."""
    if s.dtype == "category":
        __logger__.info("Skipping categorical column.")
        return s

    if lower_bound is None and upper_bound is None:
        __logger__.info("Skipping column with no boundaries.")
        return s

    if (
        lower_bound is not None
        and upper_bound is not None
        and lower_bound > upper_bound
    ):
        raise ValueError(
            f"Lower bound {lower_bound} is greater than upper bound {upper_bound}."
        )

    __logger__.info("Removing outliers from %r.", s.name)
    s = s.copy() if inplace else s

    # compute mask for values that are considered outliers
    mask = select_outliers_series(
        s,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lower_inclusive=lower_inclusive,
        upper_inclusive=upper_inclusive,
    )

    # replace outliers with NaN
    s.loc[mask] = NA

    if drop:
        __logger__.info("Dropping rows which are outliers.")
        s = s.loc[~mask]

    return s


def remove_outliers_dataframe[F: DataFrame](
    df: F,
    /,
    *,
    drop: bool = True,
    inplace: bool = False,
    lower_bound: Mapping[str, float | None],
    upper_bound: Mapping[str, float | None],
    lower_inclusive: Mapping[str, bool | None],
    upper_inclusive: Mapping[str, bool | None],
    erroron_extra_bounds: bool = False,
) -> F:
    r"""Remove outliers from a DataFrame, given boundary values."""
    __logger__.info("Removing outliers from DataFrame.")
    df = df.copy() if not inplace else df
    given_bounds = set.intersection(
        *(
            set(d.keys())
            for d in [lower_bound, upper_bound, lower_inclusive, upper_inclusive]
        )
    )

    if missing_bounds := set(df.columns) - given_bounds:
        raise ValueError(f"Columns {missing_bounds} do not have bounds!")
    if (extra_bounds := given_bounds - set(df.columns)) and erroron_extra_bounds:
        raise ValueError(f"Bounds for {extra_bounds} provided, but no such columns!")

    # compute mask for values that are considered outliers
    mask = select_outliers_dataframe(
        df,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lower_inclusive=lower_inclusive,
        upper_inclusive=upper_inclusive,
    )

    # replace outliers with NaN
    for col in df:
        df.loc[mask[col], col] = NA

    # drop rows where all columns are outliers
    if drop:
        __logger__.info("Dropping rows where all columns are outliers.")
        # FIXME: https://github.com/pandas-dev/pandas/issues/56903
        m = reduce(operator.__and__, (s for _, s in mask.items()))
        df = df.loc[~m]

    return df


# endregion auxiliary functions --------------------------------------------------------
