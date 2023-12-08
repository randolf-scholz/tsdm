#!/usr/bin/env python

from typing import Generic, Literal, TypeAlias, TypeVar, overload

X: TypeAlias = Literal["x"]
Y: TypeAlias = Literal["y"]
Z: TypeAlias = Literal["z"]
Mode = TypeVar("Mode", X, Y, Z)
modes: TypeAlias = X | Y | Z


def foo(mode: modes) -> None:
    print(f"Selected {mode=}")


foo("x")
foo("y")
foo("z")


@overload
def bar(mode: X) -> None: ...
@overload
def bar(mode: Y) -> None: ...
@overload
def bar(mode: Z) -> None: ...
def bar(mode):
    print(f"Selected {mode=}")


bar("x")
bar("y")
bar("z")


class A(Generic[Mode]):
    mode: Mode

    @overload
    def __init__(self: "A[X]", mode: X) -> None: ...
    @overload
    def __init__(self: "A[Y]", mode: Y) -> None: ...
    @overload
    def __init__(self: "A[Z]", mode: Z) -> None: ...
    def __init__(self, mode):
        print(f"Selected {mode=}")
        self.mode = mode
