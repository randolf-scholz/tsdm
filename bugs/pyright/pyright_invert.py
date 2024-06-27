from typing import Generic, TypeVar

X = TypeVar("X")
Y = TypeVar("Y")


class Encoder(Generic[X, Y]):
    def encode(self, x: X, /) -> Y: ...
    def decode(self, y: Y, /) -> X: ...


def invert_encoder(encoder: Encoder[X, Y], /) -> Encoder[Y, X]: ...


class WrappedEncoder(Encoder[X, Y]):
    encoder: Encoder[X, Y]

    def __init__(self, encoder: Encoder[X, Y], /) -> None:
        self.encoder = encoder

    def __invert__(self) -> "WrappedEncoder[Y, X]":
        return WrappedEncoder(invert_encoder(self.encoder))

    def encode(self, x: X, /) -> Y:
        return self.encoder.encode(x)

    def decode(self, y: Y, /) -> X:
        return self.encoder.decode(y)
