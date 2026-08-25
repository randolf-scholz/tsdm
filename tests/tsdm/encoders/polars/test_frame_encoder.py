r"""Tests for :class:`tsdm.encoders.polars.FrameEncoder`."""

import polars as pl
from polars.testing import assert_frame_equal

from tsdm.encoders import wrap
from tsdm.encoders.polars import FrameEncoder

TEST_FRAME = pl.DataFrame(
    {
        "run": [1, 1, 2],
        "time": [0, 1, 0],
        "value": [1, 2, 3],
        "label": ["a", "b", "a"],
    },
    schema_overrides={"value": pl.Int32},
)


def add_ten(series: pl.Series, /) -> pl.Series:
    r"""Add ten to a Series."""
    return series + 10


def subtract_ten(series: pl.Series, /) -> pl.Series:
    r"""Subtract ten from a Series."""
    return series - 10


def double(series: pl.Series, /) -> pl.Series:
    r"""Double a Series."""
    return series * 2


def halve(series: pl.Series, /) -> pl.Series:
    r"""Halve a Series."""
    return series / 2


def make_encoder() -> FrameEncoder[str]:
    r"""Construct a frame encoder with simple wrapped transforms."""
    return FrameEncoder(
        {
            "run": wrap(encoder=add_ten, decoder=subtract_ten),
            "value": wrap(encoder=double, decoder=halve),
        }
    )


def test_frame_encoder_encodes_configured_columns() -> None:
    r"""Test that configured columns are encoded and other columns pass through."""
    encoder = make_encoder()
    encoder.fit(TEST_FRAME)

    encoded = encoder.encode(TEST_FRAME)
    expected = pl.DataFrame(
        {
            "run": [11, 11, 12],
            "time": [0, 1, 0],
            "value": [2, 4, 6],
            "label": ["a", "b", "a"],
        },
        schema_overrides={"value": pl.Int32},
    )

    assert_frame_equal(encoded, expected)


def test_frame_encoder_decode_inverts_encode() -> None:
    r"""Test that decoding restores the original frame and its dtypes exactly."""
    encoder = make_encoder()
    encoder.fit(TEST_FRAME)

    encoded = encoder.encode(TEST_FRAME)
    decoded = encoder.decode(encoded)

    assert_frame_equal(TEST_FRAME, decoded)
