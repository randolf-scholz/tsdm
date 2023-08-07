#!/usr/bin/env python3


class A:
    def foo(self) -> int:
        return 0


class B(A):
    def foo(self) -> str:
        return "cheat"  # type: ignore[override]


def show(obj: A) -> int:
    return obj.foo()


x: A = B()  # no error
reveal_type(A().foo())  # int
reveal_type(B().foo())  # str
reveal_type(show(B()))  # int, no error since B ≤ₙₒₘ A ⟹ B<:A
