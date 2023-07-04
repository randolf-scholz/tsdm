#!/usr/bin/env python

from typing import Callable, TypeVar, assert_type

T = TypeVar("T")

identity: Callable[[T], T] = lambda x: x
assert_type(identity(0), int)  # ✔


def foo(func: Callable[[T], T]) -> None:
    assert_type(func(0), int)  # ✘ incompatible type "int"; expected "T"
