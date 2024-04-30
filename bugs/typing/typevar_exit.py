#!/usr/bin/env python


# https://github.com/python/mypy/issues/14502
from typing import TypeVar

from pandas import DataFrame, Index, MultiIndex, Series

T = TypeVar("T", int, str, float)
P = TypeVar("P", Index, MultiIndex, Series, DataFrame)


def foo(x: T) -> T:
    if isinstance(x, int | float):
        return x + 1

    reveal_type(x)  # str
    return x + "!"


def bar(x: P) -> P:
    if isinstance(x, Index | Series):
        return x

    reveal_type(x)  # Index, MultiIndex, Series, DataFrame
    return x
