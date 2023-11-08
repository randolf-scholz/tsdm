#!/usr/bin/env python3

from typing import Generic, Literal, TypeAlias, TypeVar, overload

MODE = TypeVar("MODE", Literal["A"], Literal["B"])
modes: TypeAlias = Literal["A", "B"]

A: TypeAlias = Literal["A"]
B: TypeAlias = Literal["B"]


class Foo(Generic[MODE]):
    # @overload
    # def __new__(cls, mode: A) -> "Foo[A]": ...
    #
    # @overload
    # def __new__(cls, mode: B) -> "Foo[B]": ...
    #
    # def __new__(cls, *args, **kwargs):
    #     return super().__new__(cls)

    @overload
    def __init__(self: "Foo[A]", mode: A) -> None: ...
    @overload
    def __init__(self: "Foo[B]", mode: B) -> None: ...

    def __init__(self, mode):
        self.mode = mode


reveal_type(Foo("A"))
reveal_type(Foo("B"))
