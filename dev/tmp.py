#!/usr/bin/env python

from typing_extensions import Never, Protocol


class A(Protocol):
    def foo(self, x: int) -> None:
        ...


class B(A):
    foo: Never
    bar: float


x: A = B()  # type checks
reveal_type(B.foo)
