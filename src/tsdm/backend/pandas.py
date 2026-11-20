r"""pandas backend implementation."""

__all__ = [
    # types
    "PandasDTypeArg",
    "PandasDType",
    "PandasType",
    # Constants
    "NA_VALUES",
    # kernels
    "cast",
    "clip",
    "copy_like",
    "drop_null",
    "false_like",
    "nanmax",
    "nanmean",
    "nanmin",
    "nanstd",
    "null_like",
    "remove_outliers",
    "scalar",
    "select_outliers",
    "strip_whitespace",
    "true_like",
    "where",
    # extras
    "aggregate_nondestructive",
    "data_overview",
    "describe",
    "get_integer_cols",
    "infer_axes",
    "is_integer_series",
]

import logging
import math
import operator
import warnings
from collections.abc import Mapping
from functools import reduce
from typing import Any, Final, Literal, Optional, overload

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from pandas import NA, DataFrame, Index, MultiIndex, NaT, Series
from pandas.core.dtypes.base import ExtensionDtype
from scipy import stats

from tsdm.types.aliases import Axis

from .util import get_outlier_options

__logger__ = logging.getLogger(__name__)

PandasDType = ExtensionDtype | np.dtype
r"""Type Alias for `pandas` dtype."""
PandasDTypeArg = str | type | PandasDType
r"""Type Alias for `pandas` dtype arguments."""
type PandasType = DataFrame | Series | Index | MultiIndex
r"""Type Alias for `pandas` objects."""
type PANDAS_TYPE = Index | Series | DataFrame
r"""A type alias for pandas objects."""

NA_VALUES: Final[frozenset[object]] = frozenset({NA, NaT})
r"""pandas missing-value sentinels `NA` and `NaT`."""


def scalar(x: Any, /, dtype: Any) -> Any:
    r"""Cast a scalar to a different dtype."""
    return Index([x]).astype(dtype).item()


def cast[P: PandasType](x: P, /, dtype: Any) -> P:
    r"""Cast a `pandas` object to a different dtype."""
    return x.astype(dtype)


def drop_null[P: PandasType](x: P, /) -> P:
    r"""Drop missing values, or rows containing missing values for a DataFrame."""
    return x.dropna()


def false_like[P: PandasType](x: P, /) -> P:
    r"""Return an all-false boolean mask with the same shape as `x`."""
    m = x.isna()
    return m ^ m


def true_like[P: PandasType](x: P, /) -> P:
    r"""Return an all-true boolean mask with the same shape as `x`."""
    m = x.isna()
    return m ^ (~m)


def infer_axes(
    x: PandasType, /, *, axis: Axis = None
) -> Literal["index", "columns"] | None:
    r"""Convert an axis specification to a pandas axis name.

    Integer axes are normalized modulo the number of dimensions. For a Series,
    every integer maps to `"index"`. For a DataFrame, even integers map to
    `"index"` and odd integers map to `"columns"`.

    `None` is returned unchanged. A sequence containing one integer is also
    accepted; other specifications raise `TypeError`.
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


def clip[P: PandasType](
    x: P, /, lower: NDArray | float | None, upper: NDArray | float | None
) -> P:
    r"""Analogue to `numpy.clip`."""
    axis = "columns" if isinstance(x, DataFrame) else "index"
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


def _where[P: PandasType](cond: Any, x: P, other: Any, /) -> P:
    r"""Apply `where` while preserving DataFrame row alignment."""
    if isinstance(x, DataFrame):
        return x.where(cond, other, axis=0)
    return x.where(cond, other)


def where[
    T: (Index, MultiIndex, Series, DataFrame),
](cond: T | Any, a: T | Any, b: T | Any, /) -> T:
    r"""Analogue to `numpy.where` with null conditions treated as false.

    At least one argument should be a pandas object; the other two can be
    array-like. If none is a pandas object, fall back to `numpy.where`.

    Nulls propagate from `a` where `cond` is true and from `b` otherwise.
    """
    mask = pd.isna(cond)
    if isinstance(cond, DataFrame | Series):
        cond = cond.mask(mask, other=False).astype(bool)
    elif np.ndim(mask) == 0:
        cond = False if mask else cond
    else:
        cond = np.asarray(
            np.where(mask, np.zeros_like(mask, dtype=bool), cond), dtype=bool
        )

    for pandas_type in (DataFrame, Series, Index):
        if isinstance(a, pandas_type):
            return _where(cond, a, b)
        if isinstance(b, pandas_type):
            assert not isinstance(cond, bool)
            return _where(~cond, b, a)
        if isinstance(cond, pandas_type):
            return _where(~cond, _where(cond, cond, b), a)
    return np.where(cond, a, b)  # type: ignore


def null_like[P: PandasType](x: P, /) -> P:
    r"""Select the input values using an all-true mask.

    Note:
        The current implementation retains the values of `x`; it does not fill
        them with nulls despite the function name.
    """
    return where(true_like(x), x, NA)


def copy_like[P: PandasType](x: ArrayLike, ref: P, /) -> P:
    r"""Create a pandas object from `x` using the type, dtype, and labels of `ref`.

    Supported references are Index, Series, and DataFrame objects. Index names,
    Series indices, and DataFrame indices and columns are taken from `ref`.
    """
    match ref:
        case Index() as idx:
            return Index(x, dtype=idx.dtype, name=idx.name)
        case Series() as s:
            return Series(x, dtype=s.dtype, index=s.index)
        case DataFrame() as df:
            return DataFrame(x, index=df.index, columns=df.columns).astype(df.dtypes)
        case _:
            raise TypeError(f"Expected {PANDAS_TYPE}, got {type(ref)}.")


def _strip_whitespace_index[I: Index](index: I, /) -> I:
    r"""Strip whitespace from all string elements in an Index."""
    if index.dtype == "string":
        return index.str.strip()
    return index


def _strip_whitespace_series[S: Series](series: S, /) -> S:
    r"""Strip whitespace from all string elements in a Series."""
    if series.dtype == "string":
        return series.str.strip()
    return series


def _strip_whitespace_dataframe[DF: DataFrame](frame: DF, /, *cols: str) -> DF:
    r"""Strip whitespace from selected columns in a DataFrame."""
    return frame.assign(
        **{col: _strip_whitespace_series(frame[col]) for col in (cols or frame)}
    )


@overload
def strip_whitespace[T: Series | Index](obj: T, /) -> T: ...
@overload
def strip_whitespace[F: DataFrame](obj: F, /, *cols: str) -> F: ...
def strip_whitespace(
    obj: Series | Index | DataFrame, /, *cols: str
) -> Series | Index | DataFrame:
    r"""Strip string values, restricting DataFrame processing to `cols` if given."""
    if cols and not isinstance(obj, DataFrame):
        raise ValueError("Cannot specify columns for a Series or Index.")
    match obj:
        case DataFrame() as df:
            return _strip_whitespace_dataframe(df, *cols)
        case Series() as s:
            return _strip_whitespace_series(s)
        case Index() as idx:
            return _strip_whitespace_index(idx)
        case _:
            raise TypeError(f"Expected {PANDAS_TYPE}, got {type(obj)}.")


def _select_outliers_series(
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


def _select_outliers_dataframe(
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
            mask[col] = _select_outliers_series(
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
    r"""Remove outliers from a Series without modifying the input."""
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
    s = s.copy()

    # compute mask for values that are considered outliers
    mask = _select_outliers_series(
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


def _remove_outliers_dataframe[F: DataFrame](
    df: F,
    /,
    *,
    drop: bool = True,
    lower_bound: Mapping[str, float | None],
    upper_bound: Mapping[str, float | None],
    lower_inclusive: Mapping[str, bool | None],
    upper_inclusive: Mapping[str, bool | None],
) -> F:
    r"""Remove outliers only from bounded columns without modifying the input.

    Each bounded column must occur in all four mappings. Unknown columns raise
    `ValueError`. With `drop=True`, drop rows where all bounded columns are outliers.
    """
    __logger__.info("Removing outliers from DataFrame.")
    df = df.copy()
    given_bounds = set.union(
        *(
            set(d.keys())
            for d in [lower_bound, upper_bound, lower_inclusive, upper_inclusive]
        )
    )

    if extra_bounds := given_bounds - set(df.columns):
        raise ValueError(f"Bounds for {extra_bounds} provided, but no such columns!")
    if not given_bounds:
        return df
    columns = [column for column in df.columns if column in given_bounds]

    # compute mask for values that are considered outliers
    mask = _select_outliers_dataframe(
        df[columns],
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lower_inclusive=lower_inclusive,
        upper_inclusive=upper_inclusive,
    )

    # replace outliers with NaN
    for col in mask:
        df.loc[mask[col], col] = NA

    # drop rows where all columns are outliers
    if drop:
        __logger__.info("Dropping rows where all columns are outliers.")
        # FIXME: https://github.com/pandas-dev/pandas/issues/56903
        m = reduce(operator.__and__, (mask[col] for col in mask.columns))
        df = df.loc[~m]

    return df


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
    that side unbounded. Limits tables must be pandas tables and identify
    columns with `variable`. Mappings may contain either the four bound names
    or per-variable boundary information. With `limits`, all boundary keywords
    must be `None`. Finite bounds require an explicit inclusivity flag.
    Limits tables may instead use their index to identify variables.
    """
    if isinstance(limits, DataFrame):
        frame = limits.set_index("variable") if "variable" in limits else limits
        limits = frame.to_dict()
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
    that side unbounded. Limits tables must be pandas tables and identify
    columns with `variable`. Mappings may contain either the four bound names
    or per-variable boundary information. With `limits`, all boundary keywords
    must be `None`. Finite bounds require an explicit inclusivity flag.
    Limits tables may instead use their index to identify variables.

    For tables, supply complete bounds for any subset of columns. Other columns
    are retained unchanged. Bounds for unknown columns raise `ValueError`.
    With `drop=True`, drop rows where every bounded column is an outlier.
    """
    if isinstance(limits, DataFrame):
        frame = limits.set_index("variable") if "variable" in limits else limits
        limits = frame.to_dict()
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


def is_integer_series(s: Series, /) -> bool:
    r"""Check if all float values are integral."""
    mask = pd.notna(s)
    return s[mask].apply(float.is_integer).all().item()


def get_integer_cols(table: DataFrame, /) -> set[str]:
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


def aggregate_nondestructive(df: DataFrame, /) -> DataFrame:
    r"""Compact simultaneous measurements while preserving each column's values.

    Given a `pandas.DataFrame` of size $m×k$, construct a new DataFrame of size
    $m′×k$, where `m′ = max(df.notna().sum())` is the largest count of non-null
    values in any column. Preserve the order of non-null values within each
    column and pad shorter columns with missing values.

    Note:
        All rows should represent simultaneous measurements. Columns are compacted
        independently, so relationships between values in the same input row are
        not preserved. The output uses the first $m′$ index labels of the input.

    Examples:
        Compact the following measurements:

        ```text
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
        ```

        The result is:

        ```text
                              Acetate  Base   DOT  Fluo_GFP   Glucose  OD600  Probe_Volume    pH
        elapsed_time
        2020-12-09 09:48:38  0.116585  <NA>  <NA>    1112.5  4.578233  0.445           200  <NA>
        2020-12-09 09:48:38  0.114842  <NA>  <NA>     912.5  4.554859  0.485          <NA>  <NA>
        ```
    """
    mask = df.notna()
    nitems = mask.sum()
    nrows = nitems.max()
    result = DataFrame(index=df.index[:nrows], columns=df.columns)
    result = result.astype(df.dtypes)

    for col in result.columns:
        result[col].iloc[: nitems[col]] = df[col].loc[mask[col]]
    return result


def describe(
    s: Series | DataFrame,
    /,
    *,
    quantiles: tuple[float, ...] = (0, 0.01, 0.5, 0.99, 1),
) -> DataFrame:
    r"""Compute summary statistics for a Series or each column of a DataFrame."""
    if isinstance(s, DataFrame):
        df = s
        return pd.concat([describe(df[col]) for col in df])

    # crete a zero-scalar of the same type as the series
    num = len(s)

    # compute the special values
    max_value = s.max()
    min_value = s.min()
    mode_value = s.mode().iloc[0]

    try:
        mean_value = s.mean()
        median_value = s.median()
        std_value = s.std()
    except Exception:
        mean_value = math.nan
        std_value = math.nan
        median_value = math.nan

    # compute counts
    max_count = (s == max_value).sum()
    min_count = (s == min_value).sum()
    mode_count = (s == mode_value).sum()
    nan_count = s.isna().sum()
    unique_count = s.nunique()

    try:
        idx = s.first_valid_index()
        # NOTE: dropna is necessary for duplicate index
        val = s.loc[idx].dropna().iloc[0] if idx is not None else 0
        zero = val - val
        neg_count = (s < zero).sum()
        pos_count = (s > zero).sum()
        zero_count = (s == zero).sum()
    except Exception:
        neg_count = NA
        pos_count = NA
        zero_count = NA

    # compute the rates
    max_rate = max_count / num
    min_rate = min_count / num
    mode_rate = mode_count / num
    nan_rate = nan_count / num
    neg_rate = neg_count / num
    pos_rate = pos_count / num
    unique_rate = unique_count / num
    zero_rate = zero_count / num

    # stats
    try:
        entropy = stats.entropy(s.value_counts(), base=2)
    except Exception:
        entropy = NA
    try:
        quantile_values = s.quantile(quantiles)
    except Exception:
        quantile_values = Series([math.nan] * len(quantiles))

    return DataFrame(
        {
            # stats
            ("stats", "entropy"): entropy,
            **{
                ("quantile", f"{q:.2f}"): v
                for q, v in zip(quantiles, quantile_values, strict=True)
            },
            # special values
            ("value", "max"): max_value,
            ("value", "mean"): mean_value,
            ("value", "median"): median_value,
            ("value", "min"): min_value,
            ("value", "mode"): mode_value,
            ("value", "std"): std_value,
            # counts
            ("count", "max"): max_count,
            ("count", "min"): min_count,
            ("count", "mode"): mode_count,
            ("count", "nan"): nan_count,
            ("count", "neg"): neg_count,
            ("count", "pos"): pos_count,
            ("count", "unique"): unique_count,
            ("count", "zero"): zero_count,
            # rates
            ("rate", "max"): max_rate,
            ("rate", "min"): min_rate,
            ("rate", "mode"): mode_rate,
            ("rate", "nan"): nan_rate,
            ("rate", "neg"): neg_rate,
            ("rate", "pos"): pos_rate,
            ("rate", "unique"): unique_rate,
            ("rate", "zero"): zero_rate,
        },
        index=[s.name],
    )


def data_overview(
    df: DataFrame, /, *, index_col: Optional[int | str] = None, digits: int = 2
) -> DataFrame:
    r"""Get a summary of the data."""
    overview = DataFrame(index=df.columns)
    null_values = df.isna()
    numerical_cols = df.select_dtypes(include="number").columns

    overview["datapoints"] = (~null_values).sum()
    overview["num_unique"] = df.nunique()
    overview["missing"] = (null_values.mean() * 100).round(2)

    overview.loc[numerical_cols, "min"] = df[numerical_cols].min()
    overview.loc[numerical_cols, "mean"] = df[numerical_cols].mean()
    overview.loc[numerical_cols, "std"] = df[numerical_cols].std()
    overview.loc[numerical_cols, "max"] = df[numerical_cols].max()

    column_dtypes = {
        "datapoints" : "int64[pyarrow]",
        "num_unique" : "int64[pyarrow]",
        "missing"    : "float64[pyarrow]",
        "min"        : "float64[pyarrow]",
        "mean"       : "float64[pyarrow]",
        "std"        : "float64[pyarrow]",
        "max"        : "float64[pyarrow]",
    }  # fmt: skip

    overview = overview.astype(column_dtypes)
    for col, dtype in column_dtypes.items():
        if pd.api.types.is_float_dtype(dtype):
            overview[col] = overview[col].round(digits)

    if index_col is not None:
        freq = {}
        for col in df.columns:
            mask = pd.notna(df[col].squeeze())
            time = df.index.get_level_values(index_col)[mask]
            freq[col] = Series(time).diff().mean()
        overview["freq"] = Series(freq)
    return overview
