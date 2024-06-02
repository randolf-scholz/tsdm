from typing import Literal, Protocol, Self, SupportsIndex, overload


class SupportsSlicing[T_co](Protocol):
    @overload
    def __getitem__(self, index: SupportsIndex, /) -> T_co: ...
    @overload
    def __getitem__(self, index: slice, /) -> Self: ...


x: SupportsSlicing[int] = [1, 2, 3, 4]  # ✅ OK
y: SupportsSlicing[Literal[1, 2, 3, 4]] = (1, 2, 3, 4)  # ❌ type error

tup = (1, 2, 3, 4)
reveal_type(tup.__getitem__)
reveal_type(tup)


class SlicingProtcol[T_co](Protocol):
    @overload
    def __call__(self, index: SupportsIndex, /) -> T_co: ...
    @overload
    def __call__(self, index: slice, /) -> Self: ...


getter: SlicingProtcol[Literal[1, 2, 3, 4]] = tup.__getitem__
