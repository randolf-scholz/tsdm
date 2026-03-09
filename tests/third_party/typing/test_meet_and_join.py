from collections.abc import Callable
from typing import Any, reveal_type


def test_max(x: tuple[int, int], y: tuple[Any, ...]) -> None:
    reveal_type(max(x, y))  # tuple[int, int]
    reveal_type(max(y, x))  # tuple[Any, ...]


type Arg[T] = Callable[[T], None]


def join[T](_x: T, _y: T, /) -> T: ...
def meet[T](_x: Arg[T], _y: Arg[T], /) -> T: ...


class X: ...


class Y: ...


def test_meet(x: Arg[X], y: Arg[Y]) -> None:
    reveal_type(meet(x, y))  # ERROR (expected: X & Y)
    reveal_type(meet(y, x))  # ERROR (expected: Y & X)


def test_meet_any(x: Arg[X], y: Arg[Any]) -> None:
    reveal_type(meet(x, y))  # X     (expected: X & Any)
    reveal_type(meet(y, x))  # Any   (expected: X & Any)


def test_meet_any2(x: Arg[Arg[X]], y: Arg[Arg[Any]]) -> None:
    reveal_type(meet(x, y))  # (X) -> None    (expected: (X -> None) & (Any -> None))
    reveal_type(meet(y, x))  # (Any) -> None  (expected: (X -> None) & (Any -> None))


def test_join(x: X, y: Y) -> None:
    reveal_type(join(x, y))  # ERROR (expected: X | Y)
    reveal_type(join(y, x))  # ERROR (expected: X | Y)


def test_join1(x: X, y: Any) -> None:
    reveal_type(join(x, y))  # X     (expected: X | Any)
    reveal_type(join(y, x))  # Any   (expected: X | Any)


def test_join2(x: Arg[X], y: Arg[Any]) -> None:
    reveal_type(join(x, y))  # (X) -> None    (expected: (X -> None) | (Any -> None))
    reveal_type(join(y, x))  # (Any) -> None  (expected: (X -> None) | (Any -> None))
