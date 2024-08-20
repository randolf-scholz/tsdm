r"""Test DtypeConverter."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tsdm.encoders.dataframe import DTypeConverter


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({
        "A": pd.to_timedelta(["1 days", "2 days", "3 days"]),
        "B": [1, 2, 3],
        "C": [1.0, 2.0, 3.0],
        "D": [1.0, 2.0, 3.0],
    }).astype({
        "A": np.dtype("timedelta64[ns]"),
        "B": np.int_,
        "C": np.float64,
        "D": np.float64,
    })


# TODO: consider trying hypothesis
def test_dtype_converter(df: pd.DataFrame) -> None:
    encoder = DTypeConverter({
        "A": "duration[ns][pyarrow]",
        "B": "Int64",
        ...: "Float64",
    })
    assert df.dtypes[0] == np.dtype("timedelta64[ns]")
    encoder.fit(df)
    encoded = encoder.encode(df)
    assert encoded.dtypes[0] == "duration[ns][pyarrow]"
    assert encoded.dtypes[1] == "Int64"
    assert encoded.dtypes[2] == "Float64"
    assert encoded.dtypes[3] == "Float64"
    assert df.dtypes[0] == np.dtype("timedelta64[ns]")
    decoded = encoder.decode(encoded)
    assert_frame_equal(df, decoded)
