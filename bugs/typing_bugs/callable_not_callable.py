#!/usr/bin/env python

from types import FunctionType
from typing import Callable


def show_annotations(func: Callable) -> None:
    if isinstance(func, FunctionType):
        print(func.__annotations__)
    else:
        print(func.__call__.__annotations__)  # ✘ error [operator]


class Foo:
    def __call__(self) -> None:
        pass


def foo() -> None:
    pass


show_annotations(foo)
show_annotations(Foo())
