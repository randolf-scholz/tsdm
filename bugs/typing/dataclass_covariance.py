from dataclasses import dataclass
from typing import Mapping, Sequence


class A:
    pass


class B(A):
    pass


@dataclass
class Foo:
    x: Sequence[str] | Mapping[str, Sequence[str]]


foo = Foo({"a": ["1"]})
