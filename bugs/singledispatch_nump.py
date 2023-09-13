#!/usr/bin/env python

from abc import ABC, abstractmethod
from functools import singledispatchmethod, wraps
from typing import Generic, TypeVar

import numpy as np
import torch

TensorVar = TypeVar("TensorVar", np.ndarray, torch.Tensor)
from tsdm.utils.decorators import wrap_func


def add_hook(func, hook):
    r"""Wrap a function with pre- and post-hooks."""

    @wraps(func)
    def _wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        hook(*args, **kwargs)  # type: ignore[misc]
        return result

    return _wrapper


class A(ABC):
    counter = 0

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.foo = add_hook(cls.foo, cls.execution_counter)

    @abstractmethod
    def foo(self, x): ...

    def execution_counter(self, *args, **kwargs):
        self.counter += 1


class B(A):
    def foo(self, x):
        print(f"It's a {type(x)}")


class C(A):
    @singledispatchmethod
    def foo(self, x, /) -> None:
        raise NotImplementedError

    @foo.register
    def _(self, x: torch.Tensor, /) -> None:
        print("It's a tensor")

    @foo.register
    def _(self, x: np.ndarray, /) -> None:
        print("It's a numpy array")


x = np.random.randn(3)
y = torch.randn(3)

obj = B()
obj.foo(x)
obj.foo(y)
print(obj.counter, flush=True)

obj = C()
obj.foo(x)
obj.foo(y)
print(obj.counter, flush=True)
