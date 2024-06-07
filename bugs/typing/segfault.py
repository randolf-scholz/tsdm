from typing import Any, Literal, Protocol, TypeVar, overload

X = TypeVar("X")
Y = TypeVar("Y")
Z = TypeVar("Z")
X2 = TypeVar("X2")
Y2 = TypeVar("Y2")


class Transform(Protocol[X, Y]):
    def __or__(
        self, other: "Transform[X2, Y2]", /
    ) -> "Transform[tuple[X, X2], tuple[Y, Y2]]": ...


@overload  # 1 arg
def chain_encoders(
    e: Transform[X, Y], /, *, simplify: Literal[True] = ...
) -> Transform[X, Y]: ...
@overload  # ≥2 args
def chain_encoders(
    *es: *tuple[Transform[Any, Y], *tuple[Transform, ...], Transform[X, Any]],
    simplify: Literal[True] = ...,
) -> Transform[X, Y]: ...
def chain_encoders(*encoders: Transform, simplify: bool = True) -> Transform:
    r"""Chain encoders."""
