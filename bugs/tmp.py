from typing import overload


@overload
def foo(x: str) -> str: ...
@overload
def foo(x: int) -> int: ...
def foo(x: str | int) -> str | int:
    return x
