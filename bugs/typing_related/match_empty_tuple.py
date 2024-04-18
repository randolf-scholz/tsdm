from typing import Any, TypeAlias

Axes: TypeAlias = Any | None | int | tuple[int, ...]


def foo(x: Axes) -> None:
    match x:
        case None:
            pass
        case int():
            pass
        case []:
            print("empty")
        case _:
            print("not empty")
