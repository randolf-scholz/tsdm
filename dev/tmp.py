#!/usr/bin/env python3

from collections.abc import Callable
from typing import Generic, TypeVar, TypeVarTuple, reveal_type

X = TypeVar("X")
Y = TypeVar("Y")
Xs = TypeVarTuple("Xs")
Ys = TypeVarTuple("Ys")


class Function(Generic[X, Y]):
    def __call__(self, x: X) -> Y: ...


class CartesianProductOfFunctions(Function[tuple[*Xs], tuple[*Ys]]):
    def __init__(self, fns: list[Callable]):
        self.fns = fns

    def __call__(self, xs: tuple[*Xs]) -> tuple[*Ys]:
        return tuple(f(x) for f, x in zip(self.fns, xs))


def identity(d: dict[tuple[*Xs], tuple[*Ys]]) -> dict[tuple[*Xs], tuple[*Ys]]:
    return d


d = {(1, 2): ("a", None, 3.14)}
reveal_type(d)
reveal_type(identity)
