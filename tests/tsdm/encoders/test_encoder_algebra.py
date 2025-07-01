import unittest
from typing import assert_type, reveal_type

from tsdm.encoders import Encoder, duplicate, wrap


class TestDuplicate(unittest.TestCase):
    encoder: Encoder[str, str] = wrap(
        encoder=lambda x: f"({x} + 1)",
        decoder=lambda x: x.lstrip("(").rstrip(" + 1)"),
    )

    def test_duplicate_zero(self) -> None:
        reduce0 = lambda x: "∅"
        duplicated_encoder = duplicate(self.encoder, 0, reduction=reduce0)

        # encode
        result = duplicated_encoder.encode("a")
        assert result == ()

        # decode
        decoded_result = duplicated_encoder.decode(result)
        assert decoded_result == "∅"

    def test_duplicate_one(self) -> None:
        reduce1 = lambda x: f"abs({x[0]})"
        duplicated_encoder = duplicate(self.encoder, 1, reduction=reduce1)

        # encode
        result = duplicated_encoder("a")
        assert result == ("(a + 1)",)

        # decode
        decoded_result = duplicated_encoder.decode(result)
        assert decoded_result == "abs(a)"

    def test_duplicate_two(self) -> None:
        reduce2 = lambda x: f"max({x[0]}, {x[1]})"
        duplicated_encoder = duplicate(self.encoder, 2, reduction=reduce2)

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
