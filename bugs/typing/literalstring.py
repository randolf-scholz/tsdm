#!/usr/bin/env python

from typing import Literal, LiteralString, Never, overload, reveal_type


@overload
def foo(x: Literal["a", "b", "c"]) -> int: ...
@overload
def foo(x: LiteralString) -> Never: ...
@overload
def foo(x: str) -> int: ...
def foo(x):
    print(x)
    if x in ("a", "b", "c"):
        print(x)
        return 1
    return len(x)


reveal_type(foo("a"))


def bar(x: str) -> None:
    z = foo(x)
    reveal_type(z)
    reveal_type(foo("a"))
    reveal_type(foo("xyx"))
    reveal_type(foo("b"))
