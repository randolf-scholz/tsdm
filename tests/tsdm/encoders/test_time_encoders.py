r"""Test time encoders."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from tsdm.encoders.time import DateTimeEncoder, TimeDeltaEncoder
from tsdm.testing import assert_arrays_equal

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

DT_TRAIN_ARRAYS = {
    "python"               : list(map(datetime.fromisoformat, DT_TRAIN_DATA)),
    "numpy"                : np.array(DT_TRAIN_DATA, dtype="datetime64[ms]"),
    "pandas-index-numpy"   : pd.Index(DT_TRAIN_DATA).astype("datetime64[ms]"),
    "pandas-index-arrow"   : pd.Index(DT_TRAIN_DATA).astype("timestamp[ms][pyarrow]"),
    "pandas-series-numpy"  : pd.Series(DT_TRAIN_DATA).astype("datetime64[ms]"),
    "pandas-series-arrow"  : pd.Series(DT_TRAIN_DATA).astype("timestamp[ms][pyarrow]"),
    "pandas-datetimeindex" : pd.DatetimeIndex(DT_TRAIN_DATA),
    "polars-series"        : pl.Series(DT_TRAIN_DATA).cast(dtype=pl.Datetime()),
    "pyarrow-array"        : pa.array(DT_TRAIN_DATA).cast(pa.timestamp("ms")),
}  # fmt: skip
r"""Example datatime data for training datetime encoders."""

DT_TEST_ARRAYS = {
    "python"               : list(map(datetime.fromisoformat, DT_TEST_DATA)),
    "numpy"                : np.array(DT_TEST_DATA, dtype="datetime64[ms]"),
    "pandas-index-numpy"   : pd.Index(DT_TEST_DATA).astype("datetime64[ms]"),
    "pandas-index-arrow"   : pd.Index(DT_TEST_DATA).astype("timestamp[ms][pyarrow]"),
    "pandas-series-numpy"  : pd.Series(DT_TEST_DATA).astype("datetime64[ms]"),
    "pandas-series-arrow"  : pd.Series(DT_TEST_DATA).astype("timestamp[ms][pyarrow]"),
    "pandas-datetimeindex" : pd.DatetimeIndex(DT_TEST_DATA),
    "polars-series"        : pl.Series(DT_TEST_DATA).cast(dtype=pl.Datetime()),
    "pyarrow-array"        : pa.array(DT_TEST_DATA).cast(pa.timestamp("ms")),
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

DT_TRAIN_ARRAYS_SPARSE = {
    "numpy"                : np.array(DT_TRAIN_DATA_SPARSE, dtype="datetime64[ms]"),
    "pandas-index-numpy"   : pd.Index(DT_TRAIN_DATA_SPARSE).astype("datetime64[ms]"),
    "pandas-index-arrow"   : pd.Index(DT_TRAIN_DATA_SPARSE).astype("timestamp[ms][pyarrow]"),
    "pandas-series-numpy"  : pd.Series(DT_TRAIN_DATA_SPARSE).astype("datetime64[ms]"),
    "pandas-series-arrow"  : pd.Series(DT_TRAIN_DATA_SPARSE).astype("timestamp[ms][pyarrow]"),
    "pandas-datetimeindex" : pd.DatetimeIndex(DT_TRAIN_DATA_SPARSE),
    "polars-series"        : pl.Series(DT_TRAIN_DATA_SPARSE).cast(dtype=pl.Datetime()),
    "pyarrow-array"        : pa.array(DT_TRAIN_DATA_SPARSE).cast(pa.timestamp("ms")),
}  # fmt: skip
r"""Example sparse datatime data for training datetime encoders."""

DT_TEST_ARRAYS_SPARSE = {
    "numpy"                : np.array(DT_TEST_DATA_SPARSE, dtype="datetime64[ms]"),
    "pandas-index-numpy"   : pd.Index(DT_TEST_DATA_SPARSE).astype("datetime64[ms]"),
    "pandas-index-arrow"   : pd.Index(DT_TEST_DATA_SPARSE).astype("timestamp[ms][pyarrow]"),
    "pandas-series-numpy"  : pd.Series(DT_TEST_DATA_SPARSE).astype("datetime64[ms]"),
    "pandas-series-arrow"  : pd.Series(DT_TEST_DATA_SPARSE).astype("timestamp[ms][pyarrow]"),
    "pandas-datetimeindex" : pd.DatetimeIndex(DT_TEST_DATA_SPARSE),
    "polars-series"        : pl.Series(DT_TEST_DATA_SPARSE).cast(dtype=pl.Datetime()),
    "pyarrow-array"        : pa.array(DT_TEST_DATA_SPARSE).cast(pa.timestamp("ms")),
}  # fmt: skip
r"""Example data for testing datetime encoders."""
# endregion datetime sparse sample data ------------------------------------------------


# region timedelta sample data ---------------------------------------------------------
TD_TRAIN_DATA = [
    timedelta(seconds=0),
    timedelta(seconds=30),
    timedelta(seconds=60),
    timedelta(seconds=90),
    timedelta(seconds=120),
    timedelta(seconds=150),
]
r"""Example timedelta data with 30s steps."""

TD_TEST_DATA = [
    timedelta(seconds=30),
    timedelta(seconds=37),
    timedelta(seconds=45),
    timedelta(seconds=46),
    timedelta(seconds=47),
]
r"""Example timedelta test data with variable steps."""

TD_TRAIN_ARRAYS = {
    "python"                : TD_TRAIN_DATA,
    "numpy"                 : np.array(TD_TRAIN_DATA, dtype="timedelta64[s]"),
    "pandas-index-numpy"    : pd.Index(TD_TRAIN_DATA).astype("timedelta64[s]"),
    "pandas-index-arrow"    : pd.Index(TD_TRAIN_DATA).astype("duration[s][pyarrow]"),
    "pandas-series-numpy"   : pd.Series(TD_TRAIN_DATA).astype("timedelta64[s]"),
    "pandas-series-arrow"   : pd.Series(TD_TRAIN_DATA).astype("duration[s][pyarrow]"),
    "pandas-timedeltaindex" : pd.TimedeltaIndex(TD_TRAIN_DATA),
    "polars-series"         : pl.Series(TD_TRAIN_DATA).cast(dtype=pl.Duration()),
    "pyarrow-array"         : pa.array(TD_TRAIN_DATA).cast(pa.duration("s")),
}  # fmt: skip
r"""Example data for training timedelta encoders."""

TD_TEST_ARRAYS = {
    "python"                : TD_TEST_DATA,
    "numpy"                 : np.array(TD_TEST_DATA, dtype="timedelta64[s]"),
    "pandas-index-numpy"    : pd.Index(TD_TEST_DATA).astype("timedelta64[s]"),
    "pandas-index-arrow"    : pd.Index(TD_TEST_DATA).astype("duration[s][pyarrow]"),
    "pandas-series-numpy"   : pd.Series(TD_TEST_DATA).astype("timedelta64[s]"),
    "pandas-series-arrow"   : pd.Series(TD_TEST_DATA).astype("duration[s][pyarrow]"),
    "pandas-timedeltaindex" : pd.TimedeltaIndex(TD_TEST_DATA),
    "polars-series"         : pl.Series(TD_TEST_DATA).cast(dtype=pl.Duration()),
    "pyarrow-array"         : pa.array(TD_TEST_DATA).cast(pa.duration("s")),
}  # fmt: skip
r"""Example data for testing timedelta encoders."""
# endregion timedelta sample data ------------------------------------------------------


# region sparse timedelta sample data --------------------------------------------------
TD_TRAIN_DATA_SPARSE = [
    None,
    timedelta(seconds=30),
    timedelta(seconds=60),
    None,
    timedelta(seconds=120),
    timedelta(seconds=150),
]
r"""Example sparse timedelta data with 30s steps."""

TD_TEST_DATA_SPARSE = [
    timedelta(seconds=30),
    timedelta(seconds=37),
    timedelta(seconds=45),
    None,
    None,
]
r"""Example sparse timedelta test data with variable steps."""

TD_TRAIN_ARRAYS_SPARSE = {
    "python"                : TD_TRAIN_DATA,
    "numpy"                 : np.array(TD_TRAIN_DATA, dtype="timedelta64[s]"),
    "pandas-index-numpy"    : pd.Index(TD_TRAIN_DATA).astype("timedelta64[s]"),
    "pandas-index-arrow"    : pd.Index(TD_TRAIN_DATA).astype("duration[s][pyarrow]"),
    "pandas-series-numpy"   : pd.Series(TD_TRAIN_DATA).astype("timedelta64[s]"),
    "pandas-series-arrow"   : pd.Series(TD_TRAIN_DATA).astype("duration[s][pyarrow]"),
    "pandas-timedeltaindex" : pd.TimedeltaIndex(TD_TRAIN_DATA),
    "polars-series"         : pl.Series(TD_TRAIN_DATA).cast(dtype=pl.Duration()),
    "pyarrow-array"         : pa.array(TD_TRAIN_DATA).cast(pa.duration("s")),
}  # fmt: skip
r"""Example sparse timedelta data for training timedelta encoders."""

TD_TEST_ARRAYS_SPARSE = {
    "python"                : TD_TEST_DATA,
    "numpy"                 : np.array(TD_TEST_DATA, dtype="timedelta64[s]"),
    "pandas-index-numpy"    : pd.Index(TD_TEST_DATA).astype("timedelta64[s]"),
    "pandas-index-arrow"    : pd.Index(TD_TEST_DATA).astype("duration[s][pyarrow]"),
    "pandas-series-numpy"   : pd.Series(TD_TEST_DATA).astype("timedelta64[s]"),
    "pandas-series-arrow"   : pd.Series(TD_TEST_DATA).astype("duration[s][pyarrow]"),
    "pandas-timedeltaindex" : pd.TimedeltaIndex(TD_TEST_DATA),
    "polars-series"         : pl.Series(TD_TEST_DATA).cast(dtype=pl.Duration()),
    "pyarrow-array"         : pa.array(TD_TEST_DATA).cast(pa.duration("s")),
}  # fmt: skip
r"""Example sparse timedelta data for testing timedelta encoders."""
# endregion sparse timedelta sample data -----------------------------------------------


@pytest.mark.parametrize("name", DT_TRAIN_ARRAYS)
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_datetime_encoder(*, name: str, sparse: bool) -> None:
    r"""Test DateTimeEncoder with different data types."""
    match name:
        case "numpy":
            pytest.xfail("https://github.com/pandas-dev/pandas/issues/58403")
        case "python" | "polars-series" | "pyarrow-array":
            pytest.xfail("Universal backend not implemented!")

    if sparse:
        train_data = DT_TRAIN_ARRAYS[name]
        test_data = DT_TEST_ARRAYS[name]
    else:
        train_data = DT_TRAIN_ARRAYS_SPARSE[name]
        test_data = DT_TEST_ARRAYS_SPARSE[name]

    encoder = DateTimeEncoder()
    encoder.fit(train_data)

    # evaluate on train data
    encoded = encoder.encode(train_data)
    decoded = encoder.decode(encoded)
    assert_arrays_equal(train_data, decoded)

    # evaluate on test data
    encoded = encoder.encode(test_data)
    decoded = encoder.decode(encoded)
    assert_arrays_equal(test_data, decoded)


@pytest.mark.parametrize("name", TD_TRAIN_ARRAYS)
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_timedelta_encoder(*, name: str, sparse: bool) -> None:
    r"""Test DateTimeEncoder with different data types."""
    match name:
        case "python" | "polars-series" | "pyarrow-array":
            pytest.xfail("Universal backend not implemented!")

    if sparse:
        train_data = TD_TRAIN_ARRAYS_SPARSE[name]
        test_data = TD_TEST_ARRAYS_SPARSE[name]
    else:
        train_data = TD_TRAIN_ARRAYS[name]
        test_data = TD_TEST_ARRAYS[name]

    encoder = TimeDeltaEncoder()
    encoder.fit(train_data)

    # evaluate on train data
    encoded = encoder.encode(train_data)
    decoded = encoder.decode(encoded)
    assert_arrays_equal(train_data, decoded)

    # evaluate on test data
    encoded = encoder.encode(test_data)
    decoded = encoder.decode(encoded)
    assert_arrays_equal(test_data, decoded)
