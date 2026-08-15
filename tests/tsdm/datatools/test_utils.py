r"""Tests for generic data utilities."""

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from tsdm.datatools import get_schema


@pytest.mark.parametrize(
    ("table", "expected_columns", "expected_dtypes", "expected_index_columns"),
    [
        (
            pd.MultiIndex.from_arrays([[1, 2], [3, 4]], names=["left", "right"]),
            ["left", "right"],
            {"left": pd.Series([1]).dtype, "right": pd.Series([1]).dtype},
            [],
        ),
        (pd.Index([1, 2], name="id"), ["id"], {"id": pd.Series([1]).dtype}, []),
        (
            pd.Series([1, 2], name="value", index=pd.Index([3, 4], name="id")),
            ["value"],
            {"value": pd.Series([1]).dtype},
            ["id"],
        ),
        (
            pd.DataFrame({"value": [1, 2]}, index=pd.Index([3, 4], name="id")),
            ["value"],
            {"value": pd.Series([1]).dtype},
            ["id"],
        ),
        (
            pa.table({"value": [1, 2]}),
            ["value"],
            {"value": pa.int64()},
            [],
        ),
        (
            pl.DataFrame({"value": [1, 2]}),
            ["value"],
            {"value": pl.Int64},
            [],
        ),
    ],
)
def test_get_schema(
    table,
    expected_columns,
    expected_dtypes,
    expected_index_columns,
) -> None:
    r"""Extract schema information from each supported table type."""
    columns, dtypes, index_columns = get_schema(table)

    assert columns == expected_columns
    assert dtypes == expected_dtypes
    assert index_columns == expected_index_columns
