r"""Tests for :class:`tsdm.encoders.FrameEncoder`."""

from pandas import DataFrame, MultiIndex
from pandas.testing import assert_frame_equal

from tsdm.encoders import FrameEncoder, wrap

TEST_FRAME = DataFrame(
    {
        "value": [1, 2, 3],
        "label": ["a", "b", "a"],
    },
    index=MultiIndex.from_tuples(
        [(1, 0), (1, 1), (2, 0)],
        names=["run", "time"],
    ),
)


def make_encoder() -> FrameEncoder[str]:
    r"""Construct a frame encoder with simple wrapped transforms."""
    return FrameEncoder(
        {
            "run": wrap(encoder=lambda x: x + 10, decoder=lambda x: x - 10),
            "value": wrap(encoder=lambda x: x * 2, decoder=lambda x: x / 2),
        }
    )


def test_frame_encoder_encodes_configured_columns() -> None:
    r"""Test that the configured index level and data column are encoded."""
    encoder = make_encoder()
    encoder.fit(TEST_FRAME)

    encoded = encoder.encode(TEST_FRAME)
    expected = DataFrame(
        {
            "value": [2, 4, 6],
            "label": ["a", "b", "a"],
        },
        index=MultiIndex.from_tuples(
            [(11, 0), (11, 1), (12, 0)],
            names=["run", "time"],
        ),
    )

    assert_frame_equal(encoded, expected)


def test_frame_encoder_decode_inverts_encode() -> None:
    r"""Test that decoding an encoded frame restores the original frame exactly."""
    encoder = make_encoder()
    encoder.fit(TEST_FRAME)

    encoded = encoder.encode(TEST_FRAME)
    decoded = encoder.decode(encoded)

    assert_frame_equal(TEST_FRAME, decoded)
