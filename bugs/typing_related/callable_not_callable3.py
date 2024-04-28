#!/usr/bin/env python

from collections.abc import Callable
from datetime import timedelta
from typing import TypeVar

from pandas import Timedelta

T = TypeVar("T", bound=Timedelta)


def get_value(obj: T | Callable[[int], T], *, step: int) -> T:
    match obj:
        case Timedelta() as td:
            return td
        case Callable() as func:
            return func(step)  # ❌ "Callable" is not callable
        case _:
            raise TypeError(f"{obj} is not callable")


def get_value2(obj: T | Callable[[int], T], *, step: int) -> T:
    if isinstance(obj, Timedelta):
        return obj
    if isinstance(obj, Callable):
        return obj(step)  # ✅
    raise TypeError
