r"""Test encoder algebra functionalities."""

import unittest

from tsdm.encoders import (
    BaseEncoder,
    Compose,
    Encoder,
    EncoderDict,
    EncoderList,
    duplicate,
    wrap,
)


def test_encoder_covariant() -> None:
    r"""Test that we can upcast to a more general type."""

    def _[X, Y](enc: BaseEncoder[X, Y]) -> Encoder[X, Y]:
        return enc

    def _upcast_any(enc: BaseEncoder) -> Encoder:
        return enc


def test_encoderlist_covariant() -> None:
    r"""Test that we can upcast to a more general type."""

    def _[X, Y](
        enc: EncoderList[X, Y, BaseEncoder[X, Y]],
    ) -> EncoderList[X, Y, Encoder[X, Y]]:
        return enc


def test_encoderdict_covariant() -> None:
    r"""Test that we can upcast to a more general type."""

    def _[X, Y, K](
        enc: EncoderDict[X, Y, K, BaseEncoder],
    ) -> EncoderDict[X, Y, K, Encoder]:
        return enc


def test_compose_covariant() -> None:
    r"""Test that we can upcast to a more general type."""

    def _[X, Y](enc: Compose[X, Y, BaseEncoder]) -> Compose[X, Y, Encoder]:
        return enc


class TestDuplicate(unittest.TestCase):
    r"""Test the duplicate class."""

    encoder: Encoder[str, str] = wrap(
        encoder=lambda x: f"({x} + 1)",
        decoder=lambda x: x.removeprefix("(").removesuffix(" + 1)"),
    )

    def test_duplicate_zero(self) -> None:
        duplicated_encoder = duplicate(self.encoder, 0, reduction=lambda _: "∅")

        # encode
        result = duplicated_encoder.encode("a")
        assert result == ()

        # decode
        decoded_result = duplicated_encoder.decode(result)
        assert decoded_result == "∅"

    def test_duplicate_one(self) -> None:
        duplicated_encoder = duplicate(
            self.encoder, 1, reduction=lambda x: f"abs({x[0]})"
        )

        # encode
        result = duplicated_encoder("a")
        assert result == ("(a + 1)",)

        # decode
        decoded_result = duplicated_encoder.decode(result)
        assert decoded_result == "abs(a)"

    def test_duplicate_two(self) -> None:
        duplicated_encoder = duplicate(
            self.encoder, 2, reduction=lambda x: f"max({x[0]}, {x[1]})"
        )

        # encode
        result = duplicated_encoder("a")
        assert result == ("(a + 1)", "(a + 1)")

        # decode
        decoded_result = duplicated_encoder.decode(result)
        assert decoded_result == "max(a, a)"

    def test_duplicate_int(self) -> None:
        result = duplicate(self.encoder, 3, reduction=" | ".join)
        duplicated_encoder = wrap(result)

        # encode
        result = duplicated_encoder("a")
        assert result == ("(a + 1)", "(a + 1)", "(a + 1)")

        # decode
        decoded_result = duplicated_encoder.decode(result)
        assert decoded_result == "a | a | a"
