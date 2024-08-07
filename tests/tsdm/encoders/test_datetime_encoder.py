r"""Test time encoders."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import polars as pl
import pytest

from tsdm.encoders.time import DateTimeEncoder
from tsdm.testing import assert_arrays_equal
from tsdm.types.protocols import NumericalTensor


def make_dtarray(data: Sequence[str | None], backend: str) -> NumericalTensor:
    match backend:
        case "numpy":
            return np.array(data, dtype="datetime64[ms]")
        case "pandas-datetimeindex":
            return pd.DatetimeIndex(data)
        case "pandas-index-arrow":
            return pd.Index(data).astype("timestamp[ms][pyarrow]")
        case "pandas-index-numpy":
            return pd.Index(data).astype("datetime64[ms]")
        case "pandas-series-arrow":
            return pd.Series(data).astype("timestamp[ms][pyarrow]")
        case "pandas-series-numpy":
            return pd.Series(data).astype("datetime64[ms]")
        case "polars-series":
            return pl.Series(data).cast(dtype=pl.Datetime())
        case _:
            raise ValueError(f"Unsupported backend: {backend}.")


BACKENDS = [
    "numpy",
    "pandas-index-arrow",
    "pandas-index-numpy",
    "pandas-series-arrow",
    "pandas-series-numpy",
    "pandas-datetimeindex",
    "polars-series",
]
r"""A list of supported backends for time encoders."""

# region datetime sample data ----------------------------------------------------------
DT_TRAIN_DATA = [
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
r"""Example datetime data in ISO 8601 format with 30s steps."""

DT_TEST_DATA = [
    "2022-01-01T08:05:30",
    "2022-01-01T09:07:00",
    "2022-01-01T10:06:30",
    "2022-01-01T10:07:00",
    "2023-01-01T10:07:30",
]
r"""Example datetime data in ISO 8601 format with 30s steps."""

DT_TRAIN_SPARSE = [
    None,
    "2022-01-01T10:00:30",
    "2022-01-01T10:01:00",
    "2022-01-01T10:01:30",
    None,
    "2022-01-01T10:02:30",
    "2022-01-01T10:03:00",
    None,
    "2022-01-01T10:04:00",
    "2022-01-01T10:04:30",
    "2022-01-01T10:05:00",
]
r"""Example datetime data in ISO 8601 format with 30s steps and missing values."""

DT_TEST_SPARSE = [
    "2022-01-01T08:05:30",
    None,
    "2022-01-01T10:06:30",
    "2022-01-01T10:07:00",
    None,
]
r"""Example datetime data in ISO 8601 format with 30s steps and missing values."""

DT_TRAIN_ARRAYS = {key: make_dtarray(DT_TRAIN_DATA, key) for key in BACKENDS}
r"""Example datatime data for training datetime encoders."""
DT_TEST_ARRAYS = {key: make_dtarray(DT_TEST_DATA, key) for key in BACKENDS}
r"""Example data for testing datetime encoders."""
DT_TRAIN_ARRAYS_SPARSE = {key: make_dtarray(DT_TRAIN_SPARSE, key) for key in BACKENDS}
r"""Example sparse datatime data for training datetime encoders."""
DT_TEST_ARRAYS_SPARSE = {key: make_dtarray(DT_TEST_SPARSE, key) for key in BACKENDS}
r"""Example data for testing datetime encoders."""
# endregion datetime sample data -------------------------------------------------------


@pytest.mark.parametrize("name", DT_TRAIN_ARRAYS)
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("rounding", [False, True], ids=["no_rounding", "rounding"])
def test_datetime_encoder(*, name: str, sparse: bool, rounding: bool) -> None:
    r"""Test DateTimeEncoder with different data types."""
    if rounding and sparse and name in {"pandas-index-arrow", "pandas-series-arrow"}:
        pytest.xfail("Overflow error: https://github.com/apache/arrow/issues/43031.")

    if sparse:
        train_data = DT_TRAIN_ARRAYS_SPARSE[name]
        test_data = DT_TEST_ARRAYS_SPARSE[name]
    else:
        train_data = DT_TRAIN_ARRAYS[name]
        test_data = DT_TEST_ARRAYS[name]

    encoder: DateTimeEncoder = DateTimeEncoder()
    encoder.fit(train_data)

    # evaluate on train data
    train_encoded = encoder.encode(train_data)
    train_decoded = encoder.decode(train_encoded)
    if rounding:
        assert encoder.backend.nanmax(abs(train_data - train_decoded)) <= encoder.unit
    else:
        assert_arrays_equal(train_data, train_decoded)

    # evaluate on test data
    test_encoded = encoder.encode(test_data)
    test_decoded = encoder.decode(test_encoded)
    if rounding:
        assert encoder.backend.nanmax(abs(test_decoded - test_decoded)) <= encoder.unit
    else:
        assert_arrays_equal(test_data, test_decoded)
