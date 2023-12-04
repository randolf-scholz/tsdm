from collections.abc import Hashable


def foo(x: Hashable) -> dict[Hashable, str]:
    return {x: str(type(x))}


foo(...)
