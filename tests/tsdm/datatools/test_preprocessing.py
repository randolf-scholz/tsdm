r"""Tests for tabular preprocessing helpers."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from tsdm.datatools.preprocessing import (
    BoundaryInformation,
    remove_outliers,
    select_outliers,
)

type OutlierInput = (
    pa.Array | pa.Table | pd.Series | pd.DataFrame | pl.Series | pl.DataFrame
)
type Expected = list[int | None] | dict[str, list[int | None]]
type OutlierMask = list[bool] | dict[str, list[bool]]

FIXED_DATA: dict[str, list[int | None]] = {
    "first": [-1, 0, 1, 2, 3, None],
    "second": [0, -3, 0, 3, 3, None],
}
SERIES_LIMITS: BoundaryInformation = {
    "lower_bound": 0,
    "upper_bound": 2,
    "lower_inclusive": True,
    "upper_inclusive": True,
}
TABLE_LIMITS: dict[str, BoundaryInformation] = {
    "first": SERIES_LIMITS,
    "second": {
        "lower_bound": -2,
        "upper_bound": 2,
        "lower_inclusive": True,
        "upper_inclusive": True,
    },
}
SERIES_EXPECTED: list[int | None] = [0, 1, 2, None]
TABLE_EXPECTED: dict[str, list[int | None]] = {
    "first": [None, 0, 1, 2, None],
    "second": [0, None, 0, None, None],
}
SERIES_MASK: list[bool] = [True, False, False, False, True, False]
TABLE_MASK: dict[str, list[bool]] = {
    "first": [True, False, False, False, True, False],
    "second": [False, True, False, True, True, False],
}
OUTLIER_CASES = [
    pytest.param(
        lambda: pa.array(FIXED_DATA["first"]),
        SERIES_LIMITS,
        SERIES_EXPECTED,
        SERIES_MASK,
        pa.Array,
        id="pyarrow-array",
    ),
    pytest.param(
        lambda: pa.table(FIXED_DATA),
        TABLE_LIMITS,
        TABLE_EXPECTED,
        TABLE_MASK,
        pa.Table,
        id="pyarrow-table",
    ),
    pytest.param(
        lambda: pd.Series(FIXED_DATA["first"], name="first"),
        SERIES_LIMITS,
        SERIES_EXPECTED,
        SERIES_MASK,
        pd.Series,
        id="pandas-series",
    ),
    pytest.param(
        lambda: pd.DataFrame(FIXED_DATA),
        TABLE_LIMITS,
        TABLE_EXPECTED,
        TABLE_MASK,
        pd.DataFrame,
        id="pandas-dataframe",
    ),
    pytest.param(
        lambda: pl.Series("first", FIXED_DATA["first"]),
        SERIES_LIMITS,
        SERIES_EXPECTED,
        SERIES_MASK,
        pl.Series,
        id="polars-series",
    ),
    pytest.param(
        lambda: pl.DataFrame(FIXED_DATA),
        TABLE_LIMITS,
        TABLE_EXPECTED,
        TABLE_MASK,
        pl.DataFrame,
        id="polars-dataframe",
    ),
]


def _normalize_values(values: Sequence[Any], /) -> list[int | None]:
    r"""Convert backend-specific null values to Python ``None``."""
    return [None if pd.isna(value) else int(value) for value in values]


def _normalize(result: OutlierInput, /) -> Expected:
    r"""Convert a supported result to a backend-independent representation."""
    match result:
        case pa.Array() as array:
            return _normalize_values(array.to_pylist())
        case pa.Table() as table:
            return {
                column: _normalize_values(values)
                for column, values in table.to_pydict().items()
            }
        case pd.Series() as series:
            return _normalize_values(series.to_list())
        case pd.DataFrame() as frame:
            return {
                column: _normalize_values(frame[column].to_list())
                for column in frame.columns
            }
        case pl.Series() as series:
            return _normalize_values(series.to_list())
        case pl.DataFrame() as frame:
            return {
                column: _normalize_values(values)
                for column, values in frame.to_dict(as_series=False).items()
            }
        case _:
            raise TypeError(f"Unsupported result type: {type(result)}")


def _normalize_mask(result: OutlierInput, /) -> OutlierMask:
    r"""Convert a supported boolean result to a Python representation."""
    match result:
        case pa.Array() as array:
            return [bool(value) for value in array.to_pylist()]
        case pa.Table() as table:
            return {
                column: [bool(value) for value in values]
                for column, values in table.to_pydict().items()
            }
        case pd.Series() as series:
            return [bool(value) for value in series.to_list()]
        case pd.DataFrame() as frame:
            return {
                column: [bool(value) for value in frame[column].to_list()]
                for column in frame.columns
            }
        case pl.Series() as series:
            return [bool(value) for value in series.to_list()]
        case pl.DataFrame() as frame:
            return {
                column: [bool(value) for value in values]
                for column, values in frame.to_dict(as_series=False).items()
            }
        case _:
            raise TypeError(f"Unsupported result type: {type(result)}")


@pytest.mark.parametrize(
    ("factory", "limits", "expected", "_mask", "result_type"), OUTLIER_CASES
)
def test_remove_outliers(
    factory: Callable[[], OutlierInput],
    limits: BoundaryInformation | Mapping[str, BoundaryInformation],
    expected: Expected,
    _mask: OutlierMask,
    result_type: type[OutlierInput],
) -> None:
    r"""Remove outliers consistently across all supported tabular backends."""
    result = remove_outliers(factory(), limits)

    assert isinstance(result, result_type)
    assert _normalize(result) == expected


@pytest.mark.parametrize(
    ("factory", "limits", "_expected", "mask", "result_type"), OUTLIER_CASES
)
def test_select_outliers(
    factory: Callable[[], OutlierInput],
    limits: BoundaryInformation | Mapping[str, BoundaryInformation],
    _expected: Expected,
    mask: OutlierMask,
    result_type: type[OutlierInput],
) -> None:
    r"""Select outliers consistently across all supported tabular backends."""
    result = select_outliers(factory(), limits)

    assert isinstance(result, result_type)
    assert _normalize_mask(result) == mask
