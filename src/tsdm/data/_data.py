r"""Utility functions that act on tabular data."""

__all__ = [
    # classes
    "BoundaryInformation",
    "BoundaryTable",
    "InlineTable",
    "MultipleBoundaryInformation",
    "Schema",
    # Functions
    "aggregate_nondestructive",
    "detect_outliers",
    "detect_outliers_dataframe",
    "detect_outliers_series",
    "float_is_int",
    "get_integer_cols",
    "joint_keys",
    "make_dataframe",
    "remove_outliers",
    "remove_outliers_dataframe",
    "remove_outliers_series",
    "strip_whitespace",
    "vlookup_uniques",
]

import logging
import operator
from collections.abc import Mapping, Sequence
from functools import reduce

import pandas as pd
from pandas import DataFrame, Index, Series
from pyarrow import Array, Table
from scipy import stats
from typing_extensions import (
    Any,
    Generic,
    NamedTuple,
    NotRequired,
    Optional,
    Required,
    TypedDict,
    overload,
)

from tsdm.backend.pandas import (
    pandas_false_like,
    strip_whitespace_dataframe,
    strip_whitespace_series,
)
from tsdm.backend.pyarrow import strip_whitespace_array, strip_whitespace_table
from tsdm.types.variables import any_var as T, pandas_var, tuple_co

__logger__: logging.Logger = logging.getLogger(__name__)


class Schema(NamedTuple):
    """Table schema."""

    shape: Optional[tuple[int, int]] = None
    """Shape of the table."""
    columns: Optional[Sequence[str]] = None
    """Column names of the table."""
    dtypes: Optional[Sequence[str]] = None
    """Data types of the columns."""


class InlineTable(TypedDict, Generic[tuple_co]):
    """A table of data in a dictionary."""

    data: Required[Sequence[tuple_co]]
    columns: NotRequired[list[str]]
    dtypes: NotRequired[list[str | type]]
    schema: NotRequired[Mapping[str, Any]]
    index: NotRequired[str | list[str]]


class BoundaryInformation(TypedDict):
    """Information about the boundaries of a single variable."""

    lower_bound: float | None
    upper_bound: float | None
    lower_inclusive: bool
    upper_inclusive: bool
    dtype: NotRequired[str]


class MultipleBoundaryInformation(TypedDict):
    """Information about the boundaries of multiple variables."""

    name: Sequence[str]
    lower_bound: Sequence[float | None]
    upper_bound: Sequence[float | None]
    lower_inclusive: Sequence[bool]
    upper_inclusive: Sequence[bool]
    dtype: NotRequired[Sequence[str]]


class BoundaryTable(TypedDict):
    """A table of boundary information, indexed by variable name."""

    lower_bound: Mapping[str, float | None]
    upper_bound: Mapping[str, float | None]
    lower_inclusive: Mapping[str, bool]
    upper_inclusive: Mapping[str, bool]
    dtype: NotRequired[Mapping[str, str]]


def joint_keys(*mappings: Mapping[T, Any]) -> set[T]:
    """Find joint keys in a collection of Mappings."""
    # NOTE: `.keys()` is necessary for working with `pandas.Series` and `pandas.DataFrame`.
    return set.intersection(*map(set, (d.keys() for d in mappings)))


def make_dataframe(
    data: Sequence[tuple[Any, ...]],
    *,
    columns: list[str] = NotImplemented,
    dtypes: list[str | type] = NotImplemented,
    schema: Mapping[str, Any] = NotImplemented,
    index: str | list[str] = NotImplemented,
) -> DataFrame:
    """Make a DataFrame from a dictionary."""
    if dtypes is not NotImplemented and schema is not NotImplemented:
        raise ValueError("Cannot specify both dtypes and schema.")

    if columns is not NotImplemented:
        cols = list(columns)
    elif schema is not NotImplemented:
        cols = list(schema)
    else:
        cols = None

    df = DataFrame.from_records(data, columns=cols)

    if dtypes is not NotImplemented:
        df = df.astype(dict(zip(df.columns, dtypes)))
    elif schema is not NotImplemented:
        df = df.astype(schema)

    if index is not NotImplemented:
        df = df.set_index(index)

    return df


@overload
def strip_whitespace(table: Array, /) -> Array: ...
@overload
def strip_whitespace(table: Table, /, *cols: str) -> Table: ...
@overload
def strip_whitespace(frame: Series, /) -> Series: ...  # type: ignore[misc]
@overload
def strip_whitespace(frame: DataFrame, /, *cols: str) -> DataFrame: ...  # type: ignore[misc]
def strip_whitespace(table, /, *cols):
    """Strip whitespace from all string columns in a table or frame."""
    match table:
        case Table() as table:
            return strip_whitespace_table(table, *cols)
        case Array() as array:
            if cols:
                raise ValueError("Cannot specify columns for an Array.")
            return strip_whitespace_array(array)
        case Series() as series:
            assert not cols
            return strip_whitespace_series(series)
        case DataFrame() as frame:
            return strip_whitespace_dataframe(frame, *cols)
        case _:
            raise TypeError(f"Unsupported type: {type(table)}")


def detect_outliers_series(
    s: Series,
    /,
    *,
    lower_bound: float | None,
    upper_bound: float | None,
    lower_inclusive: bool,
    upper_inclusive: bool,
) -> Series:
    """Detect outliers in a Series, given boundary values."""
    # detect lower-bound violations
    match lower_bound, lower_inclusive:
        case None, _:
            mask_lower = pandas_false_like(s)
        case _, True:
            mask_lower = (s < lower_bound).fillna(False)
        case _, False:
            mask_lower = (s <= lower_bound).fillna(False)
        case _:
            raise ValueError("Invalid combination of lower_bound and lower_inclusive.")

    # detect upper-bound violations
    match upper_bound, upper_inclusive:
        case None, _:
            mask_upper = pandas_false_like(s)
        case _, True:
            mask_upper = (s > upper_bound).fillna(False)
        case _, False:
            mask_upper = (s >= upper_bound).fillna(False)
        case _:
            raise ValueError("Invalid combination of upper_bound and upper_inclusive.")

    if __logger__.getEffectiveLevel() >= logging.INFO:
        lower_rate = f"{mask_lower.mean():8.3%}" if mask_lower.any() else " ------%"
        upper_rate = f"{mask_upper.mean():8.3%}" if mask_upper.any() else " ------%"
        __logger__.info(
            "%s/%s lower/upper bound violations in %r", lower_rate, upper_rate, s.name
        )

    return mask_lower | mask_upper


def detect_outliers_dataframe(
    df: DataFrame,
    /,
    *,
    lower_bound: Mapping[Any, float | None],
    upper_bound: Mapping[Any, float | None],
    lower_inclusive: Mapping[Any, bool],
    upper_inclusive: Mapping[Any, bool],
) -> DataFrame:
    """Detect outliers in a DataFrame, given boundary values."""
    given_bounds = joint_keys(
        lower_bound, upper_bound, lower_inclusive, upper_inclusive
    )
    if missing_bounds := set(df.columns) - given_bounds:
        raise ValueError(f"Columns {missing_bounds} do not have bounds!")

    mask = pandas_false_like(df)
    for col in df.columns:
        mask[col] = detect_outliers_series(
            df[col],
            lower_bound=lower_bound[col],
            upper_bound=upper_bound[col],
            lower_inclusive=lower_inclusive[col],
            upper_inclusive=upper_inclusive[col],
        )

    return mask


# region overloads ---------------------------------------------------------------------
@overload
def detect_outliers(
    obj: Series,
    description: Mapping[str, None | float | bool],
    /,
) -> Series: ...
@overload
def detect_outliers(
    obj: Series,
    /,
    *,
    lower_bound: float | None,
    upper_bound: float | None,
    lower_inclusive: bool,
    upper_inclusive: bool,
) -> Series: ...
@overload
def detect_outliers(
    obj: DataFrame,
    description: Mapping[str, Mapping[Any, None | float | bool]],
    /,
) -> DataFrame: ...
@overload
def detect_outliers(
    obj: DataFrame,
    /,
    *,
    lower_bound: Mapping[Any, float | None],
    upper_bound: Mapping[Any, float | None],
    lower_inclusive: Mapping[Any, bool],
    upper_inclusive: Mapping[Any, bool],
) -> DataFrame: ...


# endregion overloads ------------------------------------------------------------------
def detect_outliers(
    obj,
    description=None,
    /,
    *,
    lower_bound=None,
    upper_bound=None,
    lower_inclusive=None,
    upper_inclusive=None,
):
    """Detect outliers in a Series or DataFrame, given boundary values."""
    options = {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "lower_inclusive": lower_inclusive,
        "upper_inclusive": upper_inclusive,
    }

    if description is not None:
        assert all(val is None for val in options.values())
        options |= {key: description[key] for key in options}

    match obj:
        case Series() as s:
            return detect_outliers_series(s, **options)  # pyright: ignore
        case DataFrame() as df:
            return detect_outliers_dataframe(df, **options)  # pyright: ignore
        case _:
            raise TypeError(f"Unsupported type: {type(obj)}")


def remove_outliers_series(
    s: Series,
    /,
    *,
    drop: bool = True,
    inplace: bool = False,
    lower_bound: float | None,
    upper_bound: float | None,
    lower_inclusive: bool,
    upper_inclusive: bool,
) -> Series:
    """Remove outliers from a Series, given boundary values."""
    if s.dtype == "category":
        __logger__.info("Skipping categorical column.")
        return s

    if lower_bound is None and upper_bound is None:
        __logger__.info("Skipping column with no boundaries.")
        return s

    if lower_bound is not None and upper_bound is not None:
        if lower_bound > upper_bound:
            raise ValueError(
                f"Lower bound {lower_bound} is greater than upper bound {upper_bound}."
            )

    __logger__.info("Removing outliers from %r.", s.name)
    s = s.copy() if inplace else s

    # compute mask for values that are considered outliers
    mask = detect_outliers_series(
        s,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lower_inclusive=lower_inclusive,
        upper_inclusive=upper_inclusive,
    )

    # replace outliers with NaN
    s.loc[mask] = pd.NA

    if drop:
        __logger__.info("Dropping rows which are outliers.")
        s = s.loc[~mask]

    return s


def remove_outliers_dataframe(
    df: DataFrame,
    /,
    *,
    drop: bool = True,
    inplace: bool = False,
    lower_bound: Mapping[T, float | None],
    upper_bound: Mapping[T, float | None],
    lower_inclusive: Mapping[T, bool],
    upper_inclusive: Mapping[T, bool],
) -> DataFrame:
    """Remove outliers from a DataFrame, given boundary values."""
    __logger__.info("Removing outliers from DataFrame.")
    df = df.copy() if not inplace else df

    given_bounds = joint_keys(
        lower_bound, upper_bound, lower_inclusive, upper_inclusive
    )

    if missing_bounds := set(df.columns) - given_bounds:
        raise ValueError(f"Columns {missing_bounds} do not have bounds!")
    # if extra_bounds := given_bounds - set(df.columns):
    #     raise ValueError(f"Bounds for {extra_bounds} provided, but no such columns!")

    # compute mask for values that are considered outliers
    mask = detect_outliers_dataframe(
        df,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lower_inclusive=lower_inclusive,
        upper_inclusive=upper_inclusive,
    )

    # replace outliers with NaN
    for col in df:
        df.loc[mask[col], col] = pd.NA

    # drop rows where all columns are outliers
    if drop:
        __logger__.info("Dropping rows where all columns are outliers.")
        # FIXME: https://github.com/pandas-dev/pandas/issues/54389
        m = reduce(operator.__and__, (s for _, s in mask.items()))
        df = df.loc[~m]

    return df


# region overloads ---------------------------------------------------------------------
@overload
def remove_outliers(
    s: Series,
    /,
    description: BoundaryInformation,
    *,
    drop: bool = True,
    inplace: bool = False,
) -> Series: ...
@overload
def remove_outliers(
    s: Series,
    /,
    *,
    drop: bool = True,
    inplace: bool = False,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    lower_inclusive: bool = True,
    upper_inclusive: bool = True,
) -> Series: ...
@overload
def remove_outliers(
    df: DataFrame,
    /,
    description: BoundaryInformation | DataFrame,
    *,
    drop: bool = True,
    inplace: bool = False,
) -> DataFrame: ...
@overload
def remove_outliers(
    df: DataFrame,
    /,
    *,
    drop: bool = True,
    inplace: bool = False,
    lower_bound: Mapping[T, float | None],
    upper_bound: Mapping[T, float | None],
    lower_inclusive: Mapping[T, bool],
    upper_inclusive: Mapping[T, bool],
) -> DataFrame: ...


# endregion overloads ------------------------------------------------------------------
def remove_outliers(
    obj,
    description=None,
    /,
    *,
    drop=True,
    inplace=False,
    lower_bound=None,
    upper_bound=None,
    lower_inclusive=None,
    upper_inclusive=None,
):
    """Remove outliers from a DataFrame, given boundary values."""
    options = {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "lower_inclusive": lower_inclusive,
        "upper_inclusive": upper_inclusive,
    }

    if description is not None:
        assert all(val is None for val in options.values())
        options |= {key: description[key] for key in options}

    options |= {"drop": drop, "inplace": inplace}

    match obj:
        case Series() as s:
            return remove_outliers_series(s, **options)  # pyright: ignore
        case DataFrame() as df:
            return remove_outliers_dataframe(df, **options)  # pyright: ignore
        case _:
            raise TypeError(f"Expected Series or DataFrame, got {type(obj)}")


def float_is_int(series: Series) -> bool:
    r"""Check if all float values are integers."""
    mask = pd.notna(series)
    return series[mask].apply(float.is_integer).all()


def get_integer_cols(table: DataFrame) -> set[str]:
    r"""Get all columns that contain only integers."""
    cols: set[str] = set()
    for col in table.columns:
        if pd.api.types.is_integer_dtype(table[col]):
            __logger__.debug("Integer column                       : %s", col)
            cols.add(col)
        elif pd.api.types.is_float_dtype(table[col]) and float_is_int(table[col]):
            __logger__.debug("Integer column pretending to be float: %s", col)
            cols.add(col)
    return cols


def contains_no_information(df: DataFrame) -> Series:
    r"""Check if a DataFrame contains no information."""
    return df.nunique() <= 1


def vlookup_uniques(df: DataFrame, /, *, lookup_values: Series) -> dict[str, list]:
    r"""Vlookup unique values for each column in a dataframe."""
    uniques: dict[str, list] = {}
    for col in df.columns:
        mask = df[col].notna()
        uniques[col] = list(lookup_values.loc[mask].unique())
    return uniques


def aggregate_nondestructive(df: pandas_var) -> pandas_var:
    r"""Aggregate multiple simulataneous measurements in a non-destructive way.

    Given a `DataFrame` of size $m×k$, this will construct a new DataFrame of size $m'×k$,
    where `m' = max(df.notna().sum())` is the maximal number of measured not-null values.

    For example::

                              Acetate  Base   DOT  Fluo_GFP   Glucose  OD600  Probe_Volume    pH
        measurement_time
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
        measurement_time
        2020-12-09 09:48:38  0.116585  <NA>  <NA>    1112.5  4.578233  0.445           200  <NA>
        2020-12-09 09:48:38  0.114842  <NA>  <NA>     912.5  4.554859  0.485          <NA>  <NA>
    """
    if isinstance(df, Index | Series):
        return df

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
    # entropy: bool = True,
    # # special values
    # max_value: bool = True,
    # mean_value: bool = True,
    # median_value: bool = True,
    # min_value: bool = True,
    # mode_value: bool = True,
    # std_value: bool = True,
    # # counts of special values
    # max_count: bool = True,
    # min_count: bool = True,
    # nan_count: bool = True,
    # neg_count: bool = True,
    # pos_count: bool = True,
    # unique_count: bool = True,
    # zero_count: bool = True,
    # # frequencies of special values
    # max_rate: bool = True,
    # min_rate: bool = True,
    # mode_rate: bool = True,
    # nan_rate: bool = True,
    # neg_rate: bool = True,
    # pos_rate: bool = True,
    # zero_rate: bool = True,
    # unique_rate: bool = True,
    # stats
    quantiles: tuple[float, ...] = (0, 0.01, 0.5, 0.99, 1),
) -> DataFrame:
    r"""Describe a DataFrame on a per column basis."""
    if isinstance(s, DataFrame):
        df = s
        return pd.concat([describe(df[col]) for col in df])

    # crete a zero-scalar of the same type as the series
    N = len(s)

    # compute the special values
    max_value = s.max()
    min_value = s.min()
    mode_value = s.mode().iloc[0]

    try:
        mean_value = s.mean()
        median_value = s.median()
        std_value = s.std()
    except Exception:
        mean_value = float("nan")
        std_value = float("nan")
        median_value = float("nan")

    # compute counts
    max_count = (s == max_value).sum()
    min_count = (s == min_value).sum()
    mode_count = (s == mode_value).sum()
    nan_count = s.isna().sum()
    unique_count = s.nunique()

    try:
        idx = s.first_valid_index()
        # NOTE: dropna is necessary for duplicate index
        _val = s.loc[idx].dropna().iloc[0] if idx is not None else 0
        ZERO = _val - _val
        neg_count = (s < ZERO).sum()
        pos_count = (s > ZERO).sum()
        zero_count = (s == ZERO).sum()
    except Exception:
        neg_count = pd.NA
        pos_count = pd.NA
        zero_count = pd.NA

    # compute the rates
    max_rate = max_count / N
    min_rate = min_count / N
    mode_rate = mode_count / N
    nan_rate = nan_count / N
    neg_rate = neg_count / N
    pos_rate = pos_count / N
    unique_rate = unique_count / N
    zero_rate = zero_count / N

    # stats
    try:
        entropy = stats.entropy(s.value_counts(), base=2)
    except Exception:
        entropy = pd.NA
    try:
        quantile_values = s.quantile(quantiles)
    except Exception:
        quantile_values = Series([float("nan")] * len(quantiles))

    return DataFrame(
        {
            # stats
            ("stats", "entropy"): entropy,
            **{("quantile", f"{q:.2f}"): v for q, v in zip(quantiles, quantile_values)},
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
