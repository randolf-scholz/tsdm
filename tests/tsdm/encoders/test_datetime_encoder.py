r"""Test time encoders."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from tsdm.encoders.time import DateTimeEncoder
from tsdm.testing import assert_arrays_equal
from tsdm.types.protocols import NumericalTensor

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
    "2022-01-01T09:07:07",
    "2022-01-01T10:06:30",
    "2022-01-01T10:07:00",
    "2023-01-01T10:07:30",
]
r"""Example datetime data in ISO 8601 format with 30s steps."""

DT_TRAIN_ARRAYS: dict[str, NumericalTensor] = {
    "numpy"                : np.array(DT_TRAIN_DATA, dtype="datetime64[ms]"),
    "pandas-datetimeindex" : pd.DatetimeIndex(DT_TRAIN_DATA),
    "pandas-index-arrow"   : pd.Index(DT_TRAIN_DATA).astype("timestamp[ms][pyarrow]"),
    "pandas-index-numpy"   : pd.Index(DT_TRAIN_DATA).astype("datetime64[ms]"),
    "pandas-series-arrow"  : pd.Series(DT_TRAIN_DATA).astype("timestamp[ms][pyarrow]"),
    "pandas-series-numpy"  : pd.Series(DT_TRAIN_DATA).astype("datetime64[ms]"),
    "polars-series"        : pl.Series(DT_TRAIN_DATA).cast(dtype=pl.Datetime()),
}  # fmt: skip
r"""Example datatime data for training datetime encoders."""

DT_TEST_ARRAYS: dict[str, NumericalTensor] = {
    "numpy"                : np.array(DT_TEST_DATA, dtype="datetime64[ms]"),
    "pandas-datetimeindex" : pd.DatetimeIndex(DT_TEST_DATA),
    "pandas-index-arrow"   : pd.Index(DT_TEST_DATA).astype("timestamp[ms][pyarrow]"),
    "pandas-index-numpy"   : pd.Index(DT_TEST_DATA).astype("datetime64[ms]"),
    "pandas-series-arrow"  : pd.Series(DT_TEST_DATA).astype("timestamp[ms][pyarrow]"),
    "pandas-series-numpy"  : pd.Series(DT_TEST_DATA).astype("datetime64[ms]"),
    "polars-series"        : pl.Series(DT_TEST_DATA).cast(dtype=pl.Datetime()),
}  # fmt: skip
r"""Example data for testing datetime encoders."""
# endregion datetime sample data -------------------------------------------------------

# region datetime sparse sample data ---------------------------------------------------
DT_TRAIN_DATA_SPARSE = [
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

DT_TEST_DATA_SPARSE = [
    "2022-01-01T08:05:30",
    None,
    "2022-01-01T10:06:30",
    "2022-01-01T10:07:00",
    None,
]
r"""Example datetime data in ISO 8601 format with 30s steps and missing values."""

DT_TRAIN_ARRAYS_SPARSE: dict[str, NumericalTensor] = {
    "numpy"                : np.array(DT_TRAIN_DATA_SPARSE, dtype="datetime64[ms]"),
    "pandas-datetimeindex" : pd.DatetimeIndex(DT_TRAIN_DATA_SPARSE),
    "pandas-index-arrow"   : pd.Index(DT_TRAIN_DATA_SPARSE).astype("timestamp[ms][pyarrow]"),
    "pandas-index-numpy"   : pd.Index(DT_TRAIN_DATA_SPARSE).astype("datetime64[ms]"),
    "pandas-series-arrow"  : pd.Series(DT_TRAIN_DATA_SPARSE).astype("timestamp[ms][pyarrow]"),
    "pandas-series-numpy"  : pd.Series(DT_TRAIN_DATA_SPARSE).astype("datetime64[ms]"),
    "polars-series"        : pl.Series(DT_TRAIN_DATA_SPARSE).cast(dtype=pl.Datetime()),
}  # fmt: skip
r"""Example sparse datatime data for training datetime encoders."""

DT_TEST_ARRAYS_SPARSE: dict[str, NumericalTensor] = {
    "numpy"                : np.array(DT_TEST_DATA_SPARSE, dtype="datetime64[ms]"),
    "pandas-datetimeindex" : pd.DatetimeIndex(DT_TEST_DATA_SPARSE),
    "pandas-index-arrow"   : pd.Index(DT_TEST_DATA_SPARSE).astype("timestamp[ms][pyarrow]"),
    "pandas-index-numpy"   : pd.Index(DT_TEST_DATA_SPARSE).astype("datetime64[ms]"),
    "pandas-series-arrow"  : pd.Series(DT_TEST_DATA_SPARSE).astype("timestamp[ms][pyarrow]"),
    "pandas-series-numpy"  : pd.Series(DT_TEST_DATA_SPARSE).astype("datetime64[ms]"),
    "polars-series"        : pl.Series(DT_TEST_DATA_SPARSE).cast(dtype=pl.Datetime()),
}  # fmt: skip
r"""Example data for testing datetime encoders."""
# endregion datetime sparse sample data ------------------------------------------------


@pytest.mark.parametrize("name", DT_TRAIN_ARRAYS)
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_datetime_encoder(*, name: str, sparse: bool) -> None:
    r"""Test DateTimeEncoder with different data types."""
    if sparse:
        train_data = DT_TRAIN_ARRAYS[name]
        test_data = DT_TEST_ARRAYS[name]
    else:
        train_data = DT_TRAIN_ARRAYS_SPARSE[name]
        test_data = DT_TEST_ARRAYS_SPARSE[name]

    encoder: DateTimeEncoder = DateTimeEncoder()
    encoder.fit(train_data)

    # evaluate on train data
    encoded = encoder.encode(train_data)
    decoded = encoder.decode(encoded)
    assert_arrays_equal(train_data, decoded)

    # evaluate on test data
    encoded = encoder.encode(test_data)
    decoded = encoder.decode(encoded)
    assert_arrays_equal(test_data, decoded)
