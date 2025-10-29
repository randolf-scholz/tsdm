import unittest
from typing import assert_type, reveal_type

from tsdm.encoders import (
    BaseEncoder,
    Compose,
    Encoder,
    EncoderDict,
    EncoderList,
    duplicate,
    wrap,
)


def test_encoderlist_covariant() -> None:
    def _[X, Y](
        x: EncoderList[X, Y, BaseEncoder[X, Y]],
    ) -> EncoderList[X, Y, Encoder[X, Y]]:
        r"""Test that we can upcast to a more general type."""
        return x


def test_encoderdict_covariant() -> None:
    def _[X, Y, K](
        x: EncoderDict[X, Y, K, BaseEncoder],
    ) -> EncoderDict[X, Y, K, Encoder]:
        r"""Test that we can upcast to a more general type."""
        return x


def test_compose_covariant() -> None:
    def _[X, Y](x: Compose[X, Y, BaseEncoder]) -> Compose[X, Y, Encoder]:
        r"""Test that we can upcast to a more general type."""
        return x


class TestDuplicate(unittest.TestCase):
    encoder: Encoder[str, str] = wrap(
        encoder=lambda x: f"({x} + 1)",
        decoder=lambda x: x.removeprefix("(").removesuffix(" + 1)"),
    )

    def test_duplicate_zero(self) -> None:
        duplicated_encoder = duplicate(self.encoder, num=0, reduction=lambda _: "∅")

        # encode
        result = duplicated_encoder.encode("a")
        assert result == ()

        # decode
        decoded_result = duplicated_encoder.decode(result)
        assert decoded_result == "∅"

    def test_duplicate_one(self) -> None:
        duplicated_encoder = duplicate(
            self.encoder, num=1, reduction=lambda x: f"abs({x[0]})"
        )

        # encode
        result = duplicated_encoder("a")
        assert result == ("(a + 1)",)

        # decode
        decoded_result = duplicated_encoder.decode(result)
        assert decoded_result == "abs(a)"

    def test_duplicate_two(self) -> None:
        duplicated_encoder = duplicate(
            self.encoder, num=2, reduction=lambda x: f"max({x[0]}, {x[1]})"
        )

        # encode
        result = duplicated_encoder("a")
        assert result == ("(a + 1)", "(a + 1)")

        # decode
        decoded_result = duplicated_encoder.decode(result)
        assert decoded_result == "max(a, a)"

    def test_duplicate_int(self) -> None:
        reduceN = " | ".join
        result = duplicate(self.encoder, int(3), reduction=reduceN)
        duplicated_encoder = wrap(result)

        # encode
        result = duplicated_encoder("a")
        assert result == ("(a + 1)", "(a + 1)", "(a + 1)")

        # decode
        decoded_result = duplicated_encoder.decode(result)
        assert decoded_result == "a | a | a"
