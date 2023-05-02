r"""Data Utilities."""

__all__ = [
    # Functions
    "float_is_int",
    "get_integer_cols",
    "vlookup_uniques",
    "aggregate_nondestructive",
    "remove_outliers",
]

import logging

import pandas as pd
from pandas import DataFrame, Index, Series

from tsdm.types.variables import pandas_var

__logger__ = logging.getLogger(__package__)


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
