#!/usr/bin/env python

from typing import ClassVar


class FooMeta(type):
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        cls.a = 1


class Foo:
    a: ClassVar[int]

    def __init__(self):
        self.a = 2


class Bar(Foo):
    pass


x = Bar().a

if __name__ == "__main__":
    pass
