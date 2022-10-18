r"""TODO: Add module summary line.

TODO: Add module description.
"""

__all__ = [
    # Functions
    "float_is_int",
    "get_integer_cols",
    "vlookup_uniques",
    "aggregate_nondestructive",
]

import logging

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

__logger__ = logging.getLogger(__name__)


def float_is_int(series: Series) -> bool:
    r"""Check if all float values are integers."""
    mask = pd.notna(series)
    return series[mask].apply(float.is_integer).all()


def get_integer_cols(table: DataFrame) -> set[str]:
    r"""Get all columns that contain only integers."""
    cols = set()
    for col in table:
        if np.issubdtype(table[col].dtype, np.integer):
            __logger__.debug("Integer column                       : %s", col)
            cols.add(col)
        elif np.issubdtype(table[col].dtype, np.floating) and float_is_int(table[col]):
            __logger__.debug("Integer column pretending to be float: %s", col)
            cols.add(col)
    return cols


def contains_no_information(df: DataFrame) -> Series:
    r"""Check if a DataFrame contains no information."""
    return df.nunique() <= 1


def vlookup_uniques(df: DataFrame, /, values: Series) -> dict[str, list]:
    r"""Vlookup unique values for each column in a dataframe."""
    uniques = {}
    for col in df:
        mask = df[col].notna()
        uniques[col] = list(values.loc[mask].unique())
    return uniques


def aggregate_nondestructive(df: DataFrame) -> DataFrame:
    r"""Aggregate multiple simulataneous measurements in a non-destructive way.

    Given a DataFrame of size m×k, this will construct a new DataFrame of size m'×k,
    where m' = max(df.notna().sum()) is the maximal number of measured not-null values.

    Example
    -------
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

    becomes

                              Acetate  Base   DOT  Fluo_GFP   Glucose  OD600  Probe_Volume    pH
        measurement_time
        2020-12-09 09:48:38  0.116585  <NA>  <NA>    1112.5  4.578233  0.445           200  <NA>
        2020-12-09 09:48:38  0.114842  <NA>  <NA>     912.5  4.554859  0.485          <NA>  <NA>
    """
    if isinstance(df, Series):
        return df

    mask = df.notna()
    nitems = mask.sum()
    nrows = nitems.max()
    result = DataFrame(index=df.index[:nrows], columns=df.columns).astype(df.dtypes)

    for col in result:
        result[col].iloc[: nitems[col]] = df[col].loc[mask[col]]
    return result
