from typing import Self, overload


class Transform[X, Y]:
    def encode(self, x: X) -> Y: ...
    def decode(self, y: Y) -> X: ...
    def simplify(self) -> "Transform[X, Y]": ...


class Product[TupleIn: tuple, TupleOut: tuple](Transform[TupleIn, TupleOut]):
    fns: list[Transform]

    # fmt: off
    @overload  # n=0
    def __new__(cls, /) -> "Product[tuple[()], tuple[()]]": ...
    @overload  # n=1
    def __new__[X, Y](cls, fn: Transform[X, Y], /) -> "Product[tuple[X], tuple[Y]]": ...
    @overload  # n=2
    def __new__[X1, X2, Y1, Y2](cls, fn1: Transform[X1, Y1], fn2: Transform[X2, Y2], /) -> "Product[tuple[X1, X2], tuple[Y1, Y2]]": ...
    @overload  # n>2
    def __new__[X, Y](cls, /, *fns: Transform[X, Y]) -> "Product[tuple[X, ...], tuple[Y, ...]]": ...
    # fmt: on
    def __new__(cls, *fns: Transform) -> "Product[tuple, tuple]":
        return super().__new__(cls)

    def simplify(self) -> Transform[TupleIn, TupleOut]:
        # Annotated with `Transform[TupleIn, TupleOut]` instead of `Self`, as
        # additional steps may result in returning a type other than `Product`.
        fns = [fn.simplify() for fn in self.fns]
        return Product(*fns)  # <-- infers `Product[tuple[()], tuple[()]]`


def make(x: list[Transform]):
    obj = Product(*x)
    reveal_type(obj)


class Prod[TupleIn: tuple, TupleOut: tuple](Transform[TupleIn, TupleOut]):
    fns: list[Transform]

    # fmt: off
    @overload  # n=0
    def __new__(cls, *fns: *tuple[()]) -> "Prod[tuple[()], tuple[()]]": ...
    @overload  # n=1
    def __new__[X, Y](cls, *fns: *tuple[Transform[X, Y]]) -> "Prod[tuple[X], tuple[Y]]": ...
    @overload  # n=2
    def __new__[X1, X2, Y1, Y2](cls, *fns: *tuple[Transform[X1, Y1], Transform[X2, Y2]]) -> "Prod[tuple[X1, X2], tuple[Y1, Y2]]": ...
    @overload  # n>2
    def __new__[X, Y](cls, *fns: *tuple[Transform[X, Y], ...]) -> "Prod[tuple[X, ...], tuple[Y, ...]]": ...
    # fmt: on
    def __new__(cls, *fns: *tuple[Transform, ...]) -> Self:
        return object.__new__(cls)

    def simplify(self) -> Transform[TupleIn, TupleOut]:
        # Annotated with `Transform[TupleIn, TupleOut]` instead of `Self`, as
        # additional steps may result in returning a type other than `Product`.
        fns = [fn.simplify() for fn in self.fns]
        return Prod(*fns)  # <-- infers `Product[tuple[()], tuple[()]]`


@overload
def foo() -> tuple[()]: ...
@overload
def foo(x: int, /) -> tuple[int]: ...
@overload
def foo(x1: int, x2: int, /) -> tuple[int, int]: ...
@overload
def foo(*x: int) -> tuple[int, ...]: ...
def foo(*x: int) -> tuple[int, ...]:
    return x


vals: list[int] = [1, 2, 3]
reveal_type(foo(*vals))

as_tuple = tuple(vals)
reveal_type(foo(*as_tuple))
