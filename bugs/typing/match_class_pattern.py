from typing import TypeVar

from pandas import DataFrame, Series

T = TypeVar("T", Series, DataFrame)


def f(x: T) -> None:
    match x:
        case Series() | DataFrame():  # type:ignore[misc]
            pass
        case _:
            pass
