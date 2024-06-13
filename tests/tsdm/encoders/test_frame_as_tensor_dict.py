r"""Tests for `FrameAsTensorDict` encoder."""

import pytest
import torch
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from tsdm.encoders import FrameAsTensorDict

TEST_FRAME = DataFrame({
    "ID": [11, 12, 13, 15],
    "Mask": [True, False, True, False],
    "X": [1.0, 2.0, 3.0, 4.0],
    "Y": [0.1, 0.2, 0.3, 0.4],
})


@pytest.mark.parametrize("data", [TEST_FRAME, TEST_FRAME.set_index("ID")])
@pytest.mark.parametrize(
    "schema",
    [
        {"key": ["ID"], "M": ["Mask"], "Position": ["X", "Y"]},
        {"key": "ID", "M": "Mask", "Position": ["X", "Y"]},
        {"key": ["ID"], "M": ["Mask"], "Position": ...},
    ],
)
def test_frame_as_tensor_dict(data, schema):
    r"""Test `FrameAsTensorDict` encoder."""
    encoder = FrameAsTensorDict(schema)

    # test fit
    encoder.fit(data)
    assert encoder.is_fitted
    assert encoder.dtypes == {"key": None, "M": None, "Position": None}

    # test encode
    encoded = encoder.encode(data)
    assert isinstance(encoded, dict)
    assert encoded.keys() == {"key", "M", "Position"}
    assert encoded["key"].shape == (4,)
    assert encoded["Position"].shape == (4, 2)
    assert encoded["M"].shape == (4,)
    assert encoded["key"].dtype == torch.int64
    assert encoded["Position"].dtype == torch.float64
    assert encoded["M"].dtype == torch.bool

    # test decode
    decoded = encoder.decode(encoded)
    assert_frame_equal(data, decoded)


def test_frame2tensordict() -> None:
    r"""Make sure that the column order is preserved."""
    df = DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9],
        "D": [1, 2, 3],
    })
    encoder = FrameAsTensorDict(schema={"X": ["B"], "Y": ...})
    encoder.fit(df)
    encoded = encoder.encode(df)
    X = encoded["X"].numpy()
    Y = encoded["Y"].numpy()
    assert (df["B"].values == X).all(), "X should be equal to B"
    assert (df[["A", "C", "D"]].values == Y).all(), "Y should be equal to A, C, D"
