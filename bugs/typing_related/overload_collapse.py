#!/usr/bin/env python


from typing import Generic, Protocol, Self, TypeVar, overload

T = TypeVar("T")


class SupportsSpecialSub(Protocol[T]):
    @overload
    def __sub__(self, other: Self, /) -> T: ...
    @overload
    def __sub__(self, other: T, /) -> Self: ...


class Foo:
    def __sub__(self, other: Self, /) -> Self:
        return self


x: SupportsSpecialSub[Foo] = Foo()
