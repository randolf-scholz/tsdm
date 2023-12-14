"""Test DtypeConverter."""

import numpy as np
from hypothesis import given
from hypothesis.extra.pandas import column, data_frames
from pandas.testing import assert_frame_equal

from tsdm.encoders.dataframe import DTypeConverter


@given(
    data_frames(
        columns=[
            column(name="A", dtype=np.dtype("timedelta64[ns]")),
            column(name="B", dtype=np.int_),
            column(name="C", dtype=np.float_),
            column(name="D", dtype=np.float_),
        ]
    )
)
def test_dtype_converter(df):
    encoder = DTypeConverter(
        {"A": "duration[ns][pyarrow]", "B": "Int64", ...: "Float64"}
    )
    assert df.dtypes[0] == np.dtype("timedelta64[ns]")
    encoder.fit(df)
    encoded = encoder.encode(df)
    assert df.dtypes[0] == np.dtype("timedelta64[ns]")
    decoded = encoder.decode(encoded)
    assert_frame_equal(df, decoded)
