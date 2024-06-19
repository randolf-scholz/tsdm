r"""Test time encoders."""

from datetime import timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from tsdm.encoders.time import TimeDeltaEncoder
from tsdm.testing import assert_arrays_equal
from tsdm.types.protocols import NumericalTensor

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

TD_TRAIN_ARRAYS: dict[str, NumericalTensor] = {
    "numpy"                 : np.array(TD_TRAIN_DATA, dtype="timedelta64[s]"),
    "pandas-index-arrow"    : pd.Index(TD_TRAIN_DATA).astype("duration[s][pyarrow]"),
    "pandas-index-numpy"    : pd.Index(TD_TRAIN_DATA).astype("timedelta64[s]"),
    "pandas-series-arrow"   : pd.Series(TD_TRAIN_DATA).astype("duration[s][pyarrow]"),
    "pandas-series-numpy"   : pd.Series(TD_TRAIN_DATA).astype("timedelta64[s]"),
    "pandas-timedeltaindex" : pd.TimedeltaIndex(TD_TRAIN_DATA),
    "polars-series"         : pl.Series(TD_TRAIN_DATA).cast(dtype=pl.Duration()),
}  # fmt: skip
r"""Example data for training timedelta encoders."""

TD_TEST_ARRAYS: dict[str, NumericalTensor] = {
    "numpy"                 : np.array(TD_TEST_DATA, dtype="timedelta64[s]"),
    "pandas-index-arrow"    : pd.Index(TD_TEST_DATA).astype("duration[s][pyarrow]"),
    "pandas-index-numpy"    : pd.Index(TD_TEST_DATA).astype("timedelta64[s]"),
    "pandas-series-arrow"   : pd.Series(TD_TEST_DATA).astype("duration[s][pyarrow]"),
    "pandas-series-numpy"   : pd.Series(TD_TEST_DATA).astype("timedelta64[s]"),
    "pandas-timedeltaindex" : pd.TimedeltaIndex(TD_TEST_DATA),
    "polars-series"         : pl.Series(TD_TEST_DATA).cast(dtype=pl.Duration()),
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

TD_TRAIN_ARRAYS_SPARSE: dict[str, NumericalTensor] = {
    "numpy"                 : np.array(TD_TRAIN_DATA, dtype="timedelta64[s]"),
    "pandas-index-arrow"    : pd.Index(TD_TRAIN_DATA).astype("duration[s][pyarrow]"),
    "pandas-index-numpy"    : pd.Index(TD_TRAIN_DATA).astype("timedelta64[s]"),
    "pandas-series-arrow"   : pd.Series(TD_TRAIN_DATA).astype("duration[s][pyarrow]"),
    "pandas-series-numpy"   : pd.Series(TD_TRAIN_DATA).astype("timedelta64[s]"),
    "pandas-timedeltaindex" : pd.TimedeltaIndex(TD_TRAIN_DATA),
    "polars-series"         : pl.Series(TD_TRAIN_DATA).cast(dtype=pl.Duration()),
}  # fmt: skip
r"""Example sparse timedelta data for training timedelta encoders."""

TD_TEST_ARRAYS_SPARSE: dict[str, NumericalTensor] = {
    "numpy"                 : np.array(TD_TEST_DATA, dtype="timedelta64[s]"),
    "pandas-index-arrow"    : pd.Index(TD_TEST_DATA).astype("duration[s][pyarrow]"),
    "pandas-index-numpy"    : pd.Index(TD_TEST_DATA).astype("timedelta64[s]"),
    "pandas-series-arrow"   : pd.Series(TD_TEST_DATA).astype("duration[s][pyarrow]"),
    "pandas-series-numpy"   : pd.Series(TD_TEST_DATA).astype("timedelta64[s]"),
    "pandas-timedeltaindex" : pd.TimedeltaIndex(TD_TEST_DATA),
    "polars-series"         : pl.Series(TD_TEST_DATA).cast(dtype=pl.Duration()),
}  # fmt: skip
r"""Example sparse timedelta data for testing timedelta encoders."""
# endregion sparse timedelta sample data -----------------------------------------------


@pytest.mark.parametrize("name", TD_TRAIN_ARRAYS)
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_timedelta_encoder(*, name: str, sparse: bool) -> None:
    r"""Test DateTimeEncoder with different data types."""
    if sparse:
        train_data = TD_TRAIN_ARRAYS_SPARSE[name]
        test_data = TD_TEST_ARRAYS_SPARSE[name]
    else:
        train_data = TD_TRAIN_ARRAYS[name]
        test_data = TD_TEST_ARRAYS[name]

    encoder: TimeDeltaEncoder = TimeDeltaEncoder()
    encoder.fit(train_data)

    # evaluate on train data
    encoded = encoder.encode(train_data)
    decoded = encoder.decode(encoded)
    assert_arrays_equal(train_data, decoded)

    # evaluate on test data
    encoded = encoder.encode(test_data)
    decoded = encoder.decode(encoded)
    assert_arrays_equal(test_data, decoded)
