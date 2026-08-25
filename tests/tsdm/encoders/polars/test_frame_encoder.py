r"""Tests for :class:`tsdm.encoders.polars.FrameEncoder`."""

import polars as pl
import pytest
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


class FailingEncoder[T]:
    r"""Encoder that records fitting and then fails."""

    def __init__(self, name: str, calls: list[str], /) -> None:
        self.name = name
        self.calls = calls

    @property
    def params(self) -> dict[str, object]:
        return {}

    @property
    def requires_fit(self) -> bool:
        return False

    def fit(self, _: T, /) -> None:
        self.calls.append(self.name)
        raise ValueError(self.name)

    def encode(self, data: T, /) -> T:
        return data

    def decode(self, data: T, /) -> T:
        return data


def make_encoder() -> FrameEncoder[str]:
    r"""Construct a frame encoder with simple wrapped transforms."""
    return FrameEncoder(
        {
            "run": wrap(encoder=lambda x: x + 10, decoder=lambda x: x - 10),
            "value": wrap(encoder=lambda x: x * 2, decoder=lambda x: x / 2),
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


def test_frame_encoder_fit_reports_all_failures() -> None:
    r"""Test that fitting attempts every column and groups all failures."""
    calls: list[str] = []
    encoder = FrameEncoder(
        {
            "run": FailingEncoder("run", calls),
            "value": FailingEncoder("value", calls),
        }
    )

    with pytest.raises(ExceptionGroup) as exc_info:
        encoder.fit(TEST_FRAME)

    errors = exc_info.value.exceptions
    assert calls == ["run", "value"]
    assert [str(error) for error in errors] == ["run", "value"]
    assert [error.__notes__ for error in errors] == [
        ["FrameEncoder[run]: Failed to fit FailingEncoder."],
        ["FrameEncoder[value]: Failed to fit FailingEncoder."],
    ]
