"""Test time encoders."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import polars.testing
import pyarrow as pa
import pytest

from tsdm.encoders.time import DateTimeEncoder

RAW_DATA = [
    "2022-01-01T10:00:00",
    "2022-01-01T10:00:30",
    "2022-01-01T10:01:00",
    "2022-01-01T10:01:30",
    "2022-01-01T10:02:00",
    "2022-01-01T10:02:30",
    "2022-01-01T10:03:00",
    "2022-01-01T10:03:30",
    "2022-01-01T10:04:00",
    "2022-01-01T10:04:30",
    "2022-01-01T10:05:00",
]
"""Example datetime data in ISO 8601 format with 30s steps."""

DATETIME_SERIES = {
    # "python": list(map(datetime.fromisoformat, RAW_DATA)),
    # "numpy": np.array(RAW_DATA, dtype="datetime64[s]"),
    "pandas-index-numpy": pd.DatetimeIndex(RAW_DATA).astype("datetime64[s]"),
    "pandas-index-arrow": pd.DatetimeIndex(RAW_DATA).astype("timestamp[s][pyarrow]"),
    "pandas-series-numpy": pd.Series(RAW_DATA).astype("datetime64[s]"),
    "pandas-series-arrow": pd.Series(RAW_DATA).astype("timestamp[s][pyarrow]"),
    # "polars": pl.Series(RAW_DATA).cast(dtype=pl.Datetime()),
    # "pyarrow": pa.array(RAW_DATA).cast(pa.timestamp("s")),
}
"""Example data for testing datetime encoders."""


@pytest.mark.parametrize("name", DATETIME_SERIES)
def test_datetime_encoder(name: str) -> None:
    """Test DateTimeEncoder with different data types."""
    data = DATETIME_SERIES[name]

    encoder = DateTimeEncoder()
    encoder.fit(data)
    encoded = encoder.encode(data)
    decoded = encoder.decode(encoded)
    match data:
        case pd.Series():
            pd.testing.assert_series_equal(data, decoded)
        case pd.Index():
            pd.testing.assert_index_equal(data, decoded)
        case pd.DataFrame():
            pd.testing.assert_frame_equal(data, decoded)
        case np.ndarray():
            np.testing.assert_array_equal(data, decoded)
        case pl.Series():
            pl.testing.assert_series_equal(data, decoded)
        case pl.DataFrame():
            pl.testing.assert_frame_equal(data, decoded)
        case _:
            raise TypeError(f"Unsupported {type(data)=}")
