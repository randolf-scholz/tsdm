r"""Test functions for data (pandas/numpy/polars/pyarrow/etc.)."""

__all__ = [
    # functions
    "compare_dataframes",
    "get_uniques",
    "series_is_boolean",
    "series_is_int",
    "series_numeric_is_boolean",
    "series_to_boolean",
]


from typing import Optional, cast

import pandas as pd
from pandas import DataFrame, Series

from tsdm.constants import BOOLEAN_PAIRS


def get_uniques(series: Series, /, *, ignore_nan: bool = True) -> Series:
    r"""Return unique values, excluding nan."""
    if ignore_nan:
        mask = pd.notna(series)
        series = series[mask]
    return Series(series.unique())


def series_to_boolean(s: Series, uniques: Optional[Series] = None, /) -> Series:
    r"""Convert Series to nullable boolean."""
    if not pd.api.types.is_string_dtype(s):
        raise TypeError("Series must be 'string' dtype!")

    mask = pd.notna(s)
    values = get_uniques(s[mask]) if uniques is None else uniques
    mapping = next(
        set(values.str.lower()) <= bool_pair.keys() for bool_pair in BOOLEAN_PAIRS
    )
    s = s.copy()
    s[mask] = s[mask].map(mapping)
    return s.astype(pd.BooleanDtype())


def series_is_boolean(series: Series, /, *, uniques: Optional[Series] = None) -> bool:
    r"""Test if 'string' series could possibly be boolean."""
    if not pd.api.types.is_string_dtype(series):
        raise TypeError("Series must be 'string' dtype!")

    values = get_uniques(series) if uniques is None else uniques

    if len(values) == 0 or len(values) > 2:
        return False
    return any(
        set(values.str.lower()) <= bool_pair.keys() for bool_pair in BOOLEAN_PAIRS
    )


def series_numeric_is_boolean(s: Series, uniques: Optional[Series] = None, /) -> bool:
    r"""Test if 'numeric' series could possibly be boolean."""
    if not pd.api.types.is_numeric_dtype(s):
        raise TypeError("Series must be 'numeric' dtype!")

    values = get_uniques(s) if uniques is None else uniques
    if len(values) == 0 or len(values) > 2:
        return False
    return any(set(values) <= bool_pair.keys() for bool_pair in BOOLEAN_PAIRS)


def series_is_int(s: Series, uniques: Optional[Series] = None, /) -> bool:
    r"""Check whether float encoded column holds only integers."""
    if not pd.api.types.is_float_dtype(s):
        raise TypeError("Series must be 'float' dtype!")
    values = get_uniques(s) if uniques is None else uniques
    return cast(bool, values.apply(float.is_integer).all())


def compare_dataframes(given: DataFrame, reference: DataFrame) -> None:
    r"""Compare two dataframes."""
    if given.shape != reference.shape:
        raise AssertionError("Shapes must match!")
    if not given.columns.equals(reference.columns):
        raise AssertionError("Columns must match!")
    if not given.index.equals(reference.index):
        raise AssertionError("Indices must match!")
