r"""Tests for `FrameAsTensorDict` encoder."""

import pytest
from pandas import DataFrame
from pandas.testing import assert_frame_equal, assert_index_equal

from tsdm.encoders import FrameAsDict

TEST_FRAME = DataFrame({
    "ID": [11, 12, 13, 15],
    "Mask": [True, False, True, False],
    "X": [1.0, 2.0, 3.0, 4.0],
    "Y": [0.1, 0.2, 0.3, 0.4],
})


@pytest.mark.parametrize("df", [TEST_FRAME.set_index("ID")])
@pytest.mark.parametrize(
    "schema",
    [
        {"M": ["Mask"], "Position": ["X", "Y"]},
        {"M": "Mask", "Position": ["X", "Y"]},
        {"M": ["Mask"], "Position": ...},
    ],
)
def test_frame_as_tensor_dict(df: DataFrame, schema: dict) -> None:
    r"""Test `FrameAsTensorDict` encoder."""
    encoder = FrameAsDict(schema)

    # test fit
    encoder.fit(df)
    assert encoder.is_fitted

    # test encode
    encoded = encoder.encode(df)
    assert isinstance(encoded, dict)
    assert encoded.keys() == {"M", "Position"}
    assert encoded["Position"].shape == (4, 2)
    assert encoded["M"].shape == (4, 1)
    assert_index_equal(encoded["Position"].index, df.index)
    assert_index_equal(encoded["M"].index, df.index)

    # test decode
    decoded = encoder.decode(encoded)
    assert_frame_equal(df, decoded)
