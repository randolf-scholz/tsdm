r"""TODO: Add module summary line.

TODO: Add module description.
"""

__all__ = [
    # Functions
    "float_is_int",
    "get_integer_cols",
    "vlookup_uniques",
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
