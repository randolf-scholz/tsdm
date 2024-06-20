#!/usr/bin/env python

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class A:
    def foo(self, obj: T | Callable[..., T]) -> None:
        match obj:
            case Callable() as func:
                func()
