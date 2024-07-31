from typing import Self, overload


class Transform[X, Y]:
    def encode(self, x: X) -> Y: ...
    def decode(self, y: Y) -> X: ...
    def simplify(self) -> "Transform[X, Y]": ...


class Product[TupleIn: tuple, TupleOut: tuple](Transform[TupleIn, TupleOut]):
    fns: list[Transform]

    # fmt: off
    @overload  # n=0
    def __new__(cls, *fns: *tuple[()]) -> "Product[tuple[()], tuple[()]]": ...
    @overload  # n=1
    def __new__[X, Y](cls, *fns: *tuple[Transform[X, Y]]) -> "Product[tuple[X], tuple[Y]]": ...
    @overload  # n=>2
    def __new__[X, Y](cls, *fns: Transform[X, Y]) -> "Product[tuple[X, ...], tuple[Y, ...]]": ...
    # fmt: on

    def simplify(self) -> Transform[TupleIn, TupleOut]:
        cls = type(self)
        fns = [fn.simplify() for fn in self.fns]
        transform = cls(*fns)
        return transform

    def upcast(self) -> Transform[TupleIn, TupleOut]:
        return self
