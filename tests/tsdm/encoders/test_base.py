r"""Test basic encoder functionality."""

from dataclasses import FrozenInstanceError

import pytest

from tsdm.encoders import EncoderProtocol, IdentityEncoder, WrappedEncoder


class IntStringEncoder:
    r"""Simple bidirectional encoder used to exercise wrapped dispatch."""

    def encode(self, value: int, /) -> str:
        return str(value)

    def decode(self, value: str, /) -> int:
        return int(value)


def int_to_str(value: int, /) -> str:
    return str(value)


def str_to_int(value: str, /) -> int:
    return int(value)


def test_identity_encoder() -> None:
    DEMO = IdentityEncoder() >> IdentityEncoder()
    repr(DEMO)
    assert isinstance(DEMO, EncoderProtocol)


def test_wrapped_encoder_with_functions() -> None:
    encoder = WrappedEncoder(encoder=int_to_str, decoder=str_to_int)

    assert encoder.encode(1) == "1"
    assert encoder.decode("1") == 1


def test_wrapped_encoder_infers_decoder() -> None:
    wrapped = WrappedEncoder(encoder=IntStringEncoder())

    assert wrapped.encode(1) == "1"
    assert wrapped.decode("1") == 1


def test_wrapped_encoder_infers_encoder() -> None:
    wrapped = WrappedEncoder(decoder=IntStringEncoder())

    assert wrapped.encode("1") == 1
    assert wrapped.decode(1) == "1"


def test_wrapped_encoder_missing_direction() -> None:
    encoder = WrappedEncoder(encoder=int_to_str)

    assert encoder.encode(1) == "1"
    with pytest.raises(NotImplementedError):
        encoder.decode("1")


def test_wrapped_encoder_is_frozen() -> None:
    encoder = WrappedEncoder(encoder=int_to_str)
    attribute = "encoder"

    with pytest.raises(FrozenInstanceError):
        setattr(encoder, attribute, repr)
