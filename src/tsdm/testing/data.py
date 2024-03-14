"""Test functions for data (pandas/numpy/polars/pyarrow/etc.)."""

__all__ = [
    "compare_dataframes",
    "series_is_boolean",
    "series_is_int",
    "series_numeric_is_boolean",
]


from typing import Optional, cast

import pandas
from pandas import DataFrame, Series

from tsdm.constants import BOOLEAN_PAIRS
from tsdm.utils._utils import get_uniques


def series_is_boolean(series: Series, /, *, uniques: Optional[Series] = None) -> bool:
    r"""Test if 'string' series could possibly be boolean."""
    assert pandas.api.types.is_string_dtype(series), "Series must be 'string' dtype!"
    values = get_uniques(series) if uniques is None else uniques

    if len(values) == 0 or len(values) > 2:
        return False
    return any(
        set(values.str.lower()) <= bool_pair.keys() for bool_pair in BOOLEAN_PAIRS
    )


def series_numeric_is_boolean(s: Series, uniques: Optional[Series] = None, /) -> bool:
    r"""Test if 'numeric' series could possibly be boolean."""
    assert pandas.api.types.is_numeric_dtype(s), "Series must be 'numeric' dtype!"
    values = get_uniques(s) if uniques is None else uniques
    if len(values) == 0 or len(values) > 2:
        return False
    return any(set(values) <= set(bool_pair) for bool_pair in BOOLEAN_PAIRS)


def series_is_int(s: Series, uniques: Optional[Series] = None, /) -> bool:
    r"""Check whether float encoded column holds only integers."""
    assert pandas.api.types.is_float_dtype(s), "Series must be 'float' dtype!"
    values = get_uniques(s) if uniques is None else uniques
    return cast(bool, values.apply(float.is_integer).all())


def compare_dataframes(
    given: DataFrame, reference: DataFrame, *, rtol: float = 1e-5, atol: float = 1e-5
) -> None:
    """Compare two dataframes."""
    assert given.shape == reference.shape, "Shapes must match!"
    assert given.columns.equals(reference.columns), "Columns must match!"
    assert given.index.equals(reference.index), "Indices must match!"
