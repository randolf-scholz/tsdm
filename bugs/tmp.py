import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Self, assert_type


class Base[T]:
    def fit(self, data: T) -> Self:
        return self


@dataclass
class Foo[T](Base):
    item: T | float

    def fit[S](self, data: S) -> "Foo[S]":
        return self


foo = Foo(3.12)

bar = foo.fit("bar")

reveal_type(foo)

reveal_type(bar)
