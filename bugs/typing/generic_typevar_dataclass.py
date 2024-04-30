#!/usr/bin/env python

from typing import Generic, TypeVar

T = TypeVar("T", int, float)


class Baz(Generic[T]):
    def __getitem__(self, key):
        match self.s:
            case _:
                raise ValueError


b: Baz[int] = Baz()
