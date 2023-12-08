#!/usr/bin/env python

from typing import ParamSpec, Protocol, TypeVar, runtime_checkable

P = ParamSpec("P")
R = TypeVar("R")


class Func(Protocol[P, R]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


x: Func = lambda z: z

match x:
    case Func():
        print(0)
