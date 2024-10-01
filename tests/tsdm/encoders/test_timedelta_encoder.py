r"""Test time encoders."""

from collections.abc import Sequence
from datetime import timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from tsdm.encoders.time import TimeDeltaEncoder
from tsdm.testing import assert_arrays_equal
from tsdm.types.arrays import NumericalSeries


def make_tdarray(data: Sequence[timedelta | None], backend: str) -> NumericalSeries:
    match backend:
        case "numpy":
            return np.array(data, dtype="timedelta64[ms]")
        case "pandas-timedeltaindex":
            return pd.TimedeltaIndex(data)
        case "pandas-index-arrow":
            return pd.Index(data).astype("duration[ms][pyarrow]")
        case "pandas-index-numpy":
            return pd.Index(data).astype("timedelta64[ms]")
        case "pandas-series-arrow":
            return pd.Series(data).astype("duration[ms][pyarrow]")
        case "pandas-series-numpy":
            return pd.Series(data).astype("timedelta64[ms]")
        case "polars-series":
            return pl.Series(data).cast(dtype=pl.Duration())
        case _:
            raise ValueError(f"Unsupported backend: {backend}.")


BACKENDS = [
    "numpy",
    "pandas-index-arrow",
    "pandas-index-numpy",
    "pandas-series-arrow",
    "pandas-series-numpy",
    "pandas-timedeltaindex",
    "polars-series",
]
r"""A list of supported backends for time encoders."""

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

TD_TRAIN_SPARSE = [
    None,
    timedelta(seconds=30),
    timedelta(seconds=60),
    None,
    timedelta(seconds=120),
    timedelta(seconds=150),
]
r"""Example sparse timedelta data with 30s steps."""

TD_TEST_SPARSE = [
    timedelta(seconds=30),
    timedelta(seconds=37),
    timedelta(seconds=45),
    None,
    None,
]
r"""Example sparse timedelta test data with variable steps."""

TD_TRAIN_ARRAYS = {key: make_tdarray(TD_TRAIN_DATA, key) for key in BACKENDS}
r"""Example data for training timedelta encoders."""
TD_TEST_ARRAYS = {key: make_tdarray(TD_TEST_DATA, key) for key in BACKENDS}
r"""Example data for testing timedelta encoders."""
TD_TRAIN_ARRAYS_SPARSE = {key: make_tdarray(TD_TRAIN_SPARSE, key) for key in BACKENDS}
r"""Example sparse timedelta data for training timedelta encoders."""
TD_TEST_ARRAYS_SPARSE = {key: make_tdarray(TD_TEST_SPARSE, key) for key in BACKENDS}
r"""Example sparse timedelta data for testing timedelta encoders."""
# endregion timedelta sample data ------------------------------------------------------


@pytest.mark.parametrize("name", TD_TRAIN_ARRAYS)
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("rounding", [False, True], ids=["no_rounding", "rounding"])
def test_timedelta_encoder(*, name: str, sparse: bool, rounding: bool) -> None:
    r"""Test DateTimeEncoder with different data types."""
    if sparse:
        train_data = TD_TRAIN_ARRAYS_SPARSE[name]
        test_data = TD_TEST_ARRAYS_SPARSE[name]
    else:
        train_data = TD_TRAIN_ARRAYS[name]
        test_data = TD_TEST_ARRAYS[name]

    encoder: TimeDeltaEncoder = TimeDeltaEncoder(rounding=rounding)
    encoder.fit(train_data)

    # evaluate on train data
    encoded = encoder.encode(train_data)
    decoded = encoder.decode(encoded)
    if rounding:
        assert encoder.backend.nanmax(abs(train_data - decoded)) <= encoder.unit
    else:
        assert_arrays_equal(train_data, decoded)

    # evaluate on test data
    encoded = encoder.encode(test_data)
    decoded = encoder.decode(encoded)
    if rounding:
        assert encoder.backend.nanmax(abs(test_data - decoded)) <= encoder.unit
    else:
        assert_arrays_equal(test_data, decoded)
