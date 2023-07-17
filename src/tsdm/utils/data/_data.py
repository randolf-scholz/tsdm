r"""Data Utilities."""

__all__ = [
    # Functions
    "aggregate_nondestructive",
    "compute_entropy",
    "float_is_int",
    "get_integer_cols",
    "remove_outliers",
    "strip_whitespace",
    "table_info",
    "vlookup_uniques",
]

import logging
from typing import overload

import pandas as pd
import pyarrow as pa
import scipy.stats as stats
from pandas import DataFrame, Index, Series
from tqdm.autonotebook import tqdm

from tsdm.types.variables import pandas_var

__logger__ = logging.getLogger(__package__)


@overload
def strip_whitespace(table: pa.Array) -> pa.Array:
    ...


@overload
def strip_whitespace(table: pa.Table) -> pa.Table:  # type: ignore[misc]
    ...


@overload
def strip_whitespace(frame: Series) -> Series:
    ...


@overload
def strip_whitespace(frame: DataFrame) -> DataFrame:  # type: ignore[misc]
    ...


def strip_whitespace(table):
    """Strip whitespace from all string columns in a table or frame."""
    if isinstance(table, pa.Table):
        return _strip_whitespace_table(table)
    if isinstance(table, pa.Array):
        return _strip_whitespace_array(table)
    if isinstance(table, DataFrame):
        return _strip_whitespace_frame(table)
    if isinstance(table, Series):
        return _strip_whitespace_series(table)
    raise TypeError(f"Unsupported type: {type(table)}")


def _strip_whitespace_array(arr: pa.Array) -> pa.Array:
    if isinstance(arr, pa.Array) and arr.type == pa.string():
        return pa.compute.strip(arr)
    if isinstance(arr, pa.ListArray) and arr.type.value_type == pa.string():
        return pa.compute.map(
            pa.compute.strip,
            arr,
        )
    if isinstance(arr, pa.DictionaryArray) and arr.type.value_type == pa.string():
        return pa.DictionaryArray.from_arrays(
            arr.indices,
            pa.compute.strip(arr.dictionary),
        )
    return arr


def _strip_whitespace_table(table: pa.Table) -> pa.Table:
    for col in table.column_names:
        table = table.set_column(
            table.column_names.index(col),
            col,
            _strip_whitespace_array(table[col]),
        )
    return table


def _strip_whitespace_frame(frame: DataFrame) -> DataFrame:
    return frame.apply(_strip_whitespace_series)


def _strip_whitespace_series(series: Series) -> Series:
    if series.dtype == "string":
        return series.str.strip()
    return series


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


def remove_outliers(
    df: DataFrame, /, description: DataFrame, dropna: bool = True
) -> DataFrame:
    """Remove outliers from a DataFrame, given boundary values."""
    if mismatch := set(df.columns) ^ set(description.index):
        raise ValueError(f"Columns of data and description do not match {mismatch}.")

    boundary_cols = ["lower", "upper", "lower_included", "upper_included"]

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
