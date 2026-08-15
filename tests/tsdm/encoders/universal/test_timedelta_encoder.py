r"""Test time encoders."""

from collections.abc import Sequence
from datetime import timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from tests.test_utils.assertions import assert_arrays_equal
from tsdm.encoders import TimeDeltaEncoder


def make_tdarray(data: Sequence[timedelta | None], backend: str):
    match backend:
        case "numpy":
            return np.array(data, dtype="timedelta64[ms]")
        case "pandas-timedeltaindex":
            return pd.TimedeltaIndex(data)
        case "pandas[arrow]-index":
            return pd.Index(data).astype("duration[ms][pyarrow]")
        case "pandas[numpy]-index":
            return pd.Index(data).astype("timedelta64[ms]")
        case "pandas[arrow]-series":
            return pd.Series(data).astype("duration[ms][pyarrow]")
        case "pandas[numpy]-series":
            return pd.Series(data).astype("timedelta64[ms]")
        case "polars-series":
            return pl.Series(data).cast(dtype=pl.Duration())
        case _:
            raise ValueError(f"Unsupported backend: {backend}.")


BACKENDS = [
    "numpy",
    "pandas[arrow]-index",
    "pandas[numpy]-index",
    "pandas[arrow]-series",
    "pandas[numpy]-series",
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


@pytest.mark.parametrize("rounding", [False, True], ids=["no_rounding", "rounding"])
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("case", TD_TRAIN_ARRAYS)
def test_timedelta_encoder(case, *, sparse: bool, rounding: bool) -> None:
    r"""Test DateTimeEncoder with different data types."""
    if sparse:
        train_data = TD_TRAIN_ARRAYS_SPARSE[case]
        test_data = TD_TEST_ARRAYS_SPARSE[case]
    else:
        train_data = TD_TRAIN_ARRAYS[case]
        test_data = TD_TEST_ARRAYS[case]

    encoder: TimeDeltaEncoder = TimeDeltaEncoder(rounding=rounding)
    encoder.fit(train_data)

    # evaluate on train data
    encoded = encoder.encode(train_data)
    decoded = encoder.decode(encoded)

    assert type(decoded) is type(train_data)
    assert decoded.dtype == train_data.dtype

    if rounding:
        assert encoder.backend.nanmax(abs(train_data - decoded)) <= encoder.unit
    else:
        assert_arrays_equal(train_data, decoded)

    # evaluate on test data
    encoded = encoder.encode(test_data)
    decoded = encoder.decode(encoded)

    assert type(decoded) is type(test_data)
    assert decoded.dtype == test_data.dtype

    if rounding:
        assert encoder.backend.nanmax(abs(test_data - decoded)) <= encoder.unit
    else:
        assert_arrays_equal(test_data, decoded)
