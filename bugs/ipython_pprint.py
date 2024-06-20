#!/usr/bin/env ipython
from functools import partialmethod
from typing import NamedTuple

from IPython.display import display


def my_pprint(obj, **options):
    return "Hello World"


def add_custom_pprint(cls=None, **options):
    if cls is None:

        def decorator(cls):
            return add_custom_pprint(cls, **options)

        return decorator

    cls.__repr__ = my_pprint
    return cls


@add_custom_pprint
class MyNamedTuple(NamedTuple):
    x: int
    y: int


x = MyNamedTuple(1, 2)
print(x)  # "Hello World"
display(x)  # (1, 2)
