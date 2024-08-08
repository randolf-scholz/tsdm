from collections.abc import Sequence
from typing import Protocol, Self, overload


class Encoder(Protocol):
    def __call__(self, u, /): ...

    def simplify(self) -> "Encoder":
        return self


class Chain(Encoder, Sequence[Encoder]):
    encoders: list[Encoder]

    def __init__(self, *encoders: Encoder):
        self.encoders = list(encoders)

    def __len__(self) -> int:
        return len(self.encoders)

    @overload
    def __getitem__(self, index: int) -> Encoder: ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    def __getitem__(self, index, /):
        return self.encoders[index]

    def __call__(self, x):
        for encoder in self.encoders:
            x = encoder(x)
        return x

    def simplify(self) -> Encoder:
        match self:
            case [encoder]:
                return encoder.simplify()
            case _:
                cls = type(self)
                return cls(*(e.simplify() for e in self.encoders))

        # The equivalent code below works...
        # if len(self)==1:
        #     return self.encoders[0].simplify()
        # cls = type(self)
        # return cls(*(e.simplify() for e in self.encoders))
