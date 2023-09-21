#!/usr/bin/env python

from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeAlias, TypeVar

T = TypeVar("T")


class Foo(ABC, Generic[T]):
    @abstractmethod
    def f(self, x: T) -> str: ...


class Bar(Foo[KEY := Literal["A", "B", "C"]]):  # ✘ raises [misc, valid-type]
    assert (
        KEY == Literal["A", "B", "C"]
    )  # ✘  Cannot determine type of "KEY"  [has-type]

    def f(self, x: KEY) -> str:
        return f"The key is {x=}."


print(Bar().f("A"))  # ✔
print(Bar().f("XXX"))  # ✘ no error raised
