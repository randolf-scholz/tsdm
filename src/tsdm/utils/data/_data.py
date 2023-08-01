r"""Utility function that act on tabular data."""

__all__ = [
    # classes
    "InlineTable",
    # Functions
    "aggregate_nondestructive",
    "compute_entropy",
    "float_is_int",
    "get_integer_cols",
    "make_dataframe",
    "detect_outliers_series",
    "remove_outliers",
    "remove_outlieres_series",
    "remove_outliers_dataframe",
    "strip_whitespace",
    "table_info",
    "vlookup_uniques",
]

import logging
import operator
from collections.abc import Mapping, Sequence
from functools import reduce
from typing import Any, Generic, overload

import pandas as pd
import pyarrow as pa
from pandas import DataFrame, Index, Series
from scipy import stats
from tqdm.autonotebook import tqdm
from typing_extensions import NotRequired, Required, TypedDict

from tsdm.backend.pandas import strip_whitespace_dataframe, strip_whitespace_series
from tsdm.backend.pyarrow import strip_whitespace_array, strip_whitespace_table
from tsdm.backend.universal import false_like
from tsdm.types.variables import pandas_var, tuple_co

__logger__ = logging.getLogger(__package__)


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
def strip_whitespace(table: pa.Array, /) -> pa.Array:
    ...


@overload
def strip_whitespace(table: pa.Table, /) -> pa.Table:  # type: ignore[misc]
    ...


@overload
def strip_whitespace(frame: Series, /) -> Series:  # type: ignore[misc]
    ...


@overload
def strip_whitespace(frame: DataFrame, /) -> DataFrame:  # type: ignore[misc]
    ...


def strip_whitespace(table, /):
    """Strip whitespace from all string columns in a table or frame."""
    match table:
        case pa.Table() as table:
            return strip_whitespace_table(table)
        case pa.Array() as array:
            return strip_whitespace_array(array)
        case Series() as series:  # type: ignore[misc]
            return strip_whitespace_series(series)  # type: ignore[unreachable]
        case DataFrame() as frame:  # type: ignore[misc]
            return strip_whitespace_dataframe(frame)  # type: ignore[unreachable]
        case _:
            raise TypeError(f"Unsupported type: {type(table)}")


def compute_entropy(value_counts: pa.Array) -> float:
    r"""Compute the normalized entropy using a value_counts array.

    .. math:: ∑_{i=1}^n -pᵢ \log₂(pᵢ)/\log₂(n)

    Note:
        Since entropy is maximized for a uniform distribution, and the entropy
        of a uniform distribution of n choices is log₂(n), the normalization
        ensures that the entropy is in the range [0, 1].
    """
    counts = pa.compute.struct_field(value_counts, 1)
    n = len(counts)
    freqs = pa.compute.divide(
        pa.compute.cast(counts, pa.float64()),
        pa.compute.sum(counts),
    )

    H = pa.compute.divide(
        pa.compute.sum(pa.compute.multiply(freqs, pa.compute.log2(freqs))),
        pa.compute.log2(n),
    )
    return -H.as_py()


def detect_outliers_series(
    s: Series,
    *,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    lower_inclusive: bool = True,
    upper_inclusive: bool = True,
) -> Series:
    """Detect outliers in a Series, given boundary values."""
    mask = false_like(s)
    if lower_bound is not None:
        if lower_inclusive:
            mask |= s < lower_bound
        else:
            mask |= s <= lower_bound

    if upper_bound is not None:
        if upper_inclusive:
            mask |= s > upper_bound
        else:
            mask |= s >= upper_bound
    return mask


def remove_outlieres_series(
    s: Series,
    *,
    drop: bool = True,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    lower_inclusive: bool = True,
    upper_inclusive: bool = True,
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

    __logger__.info("Removing outliers from column.")

    # mask for values that are considered outliers
    mask = detect_outliers_series(
        s,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lower_inclusive=lower_inclusive,
        upper_inclusive=upper_inclusive,
    )
    if drop:
        s = s.loc[~mask]
    else:
        s.loc[mask] = None

    return s


def remove_outliers_dataframe(
    df: DataFrame,
    *,
    drop: bool = True,
    lower_bound: Mapping[str, float | None],
    upper_bound: Mapping[str, float | None],
    lower_inclusive: Mapping[str, bool],
    upper_inclusive: Mapping[str, bool],
) -> DataFrame:
    """Remove outliers from a DataFrame, given boundary values."""
    masks_per_col = []
    for col in df:
        mask = detect_outliers_series(
            df[col],
            lower_bound=lower_bound[col],
            upper_bound=upper_bound[col],
            lower_inclusive=lower_inclusive[col],
            upper_inclusive=upper_inclusive[col],
        )
        masks_per_col.append(mask)

        df.loc[mask, col] = None

    if drop:
        # union of all masks
        mask = reduce(operator.__or__, masks_per_col)
        df = df.loc[~mask]

    return df


def remove_outliers(
    df: DataFrame,
    /,
    description: DataFrame,
    *,
    dropna: bool = True,
) -> DataFrame:
    """Remove outliers from a DataFrame, given boundary values."""
    boundary_cols = ["lower_bound", "upper_bound", "lower_included", "upper_included"]

    if mismatch := set(df.columns) ^ set(description.index):
        raise ValueError(f"Columns of data and description do not match {mismatch}.")

    if missing_columns := set(boundary_cols) - set(description.columns):
        raise ValueError(f"Description table is missing columns {missing_columns}!")

    __logger__.info("Removing outliers from table.")
    M = max(map(len, df.columns)) + 1

    for col in df:
        s = df[col]
        if s.dtype == "category":
            __logger__.info(f"%-{M}s: Skipping categorical column.", col)
            continue

        lower, upper, lbi, ubi = description.loc[col, boundary_cols]
        if lbi:
            mask = (s < lower).fillna(False)
        else:
            mask = (s <= lower).fillna(False)
        if ubi:
            mask |= (s > upper).fillna(False)
        else:
            mask |= (s >= upper).fillna(False)
        if mask.any():
            __logger__.info(
                f"%-{M}s: Dropping %8.4f%% outliers.", col, mask.mean() * 100
            )
        else:
            __logger__.info(f"%-{M}s: No outliers detected.", col)
        df.loc[mask, col] = None
    if dropna:
        df = df.dropna(how="all", axis="index")
    return df


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


def vlookup_uniques(df: DataFrame, /, values: Series) -> dict[str, list]:
    r"""Vlookup unique values for each column in a dataframe."""
    uniques: dict[str, list] = {}
    for col in df.columns:
        mask = df[col].notna()
        uniques[col] = list(values.loc[mask].unique())
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

    return pd.DataFrame(
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


def table_info(table: pa.Table) -> None:
    """Print information about a table."""
    size = table.nbytes / (1024 * 1024 * 1024)
    print(f"shape={table.shape}  {size=:.3f} GiB")
    M = max(map(len, table.column_names)) + 1
    for name, col in tqdm(zip(table.column_names, table.columns)):
        num_total = len(col)
        num_null = pa.compute.sum(pa.compute.is_null(col)).as_py()
        value_counts = col.value_counts()
        num_uniques = len(value_counts) - bool(num_null)
        nulls = f"{num_null / num_total:7.2%}" if num_null else f"{'None':7s}"
        uniques = (
            num_uniques / (num_total - num_null)
            if num_total > num_null
            else num_uniques / num_total
        )
        entropy = compute_entropy(value_counts)
        dtype = str(col.type)[:10]
        print(
            f"{name:{M}s}  {nulls=:s}  {num_uniques=:9d} ({uniques:7.2%})"
            f"  {entropy=:7.2%}  {dtype=:s}"
        )
