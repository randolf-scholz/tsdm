#!/usr/bin/env python

from typing import Any, Generic, Protocol, Sequence, TypeVar, assert_type

import numpy as np
from numpy.typing import NDArray


class Unknown(Protocol):
    """Dummy protocol."""


T = TypeVar("T", bound=Unknown)


class Foo(Generic[T]):
    """Dummy class."""

    data: NDArray[T]

    def __init__(self, data: Sequence[T]):
        self.data = np.array(data)

    def where_greater_zero(self) -> NDArray[T]:
        zero = self.data[0] * 0
        return self.data[self.data > zero]


obj: Foo[Any] = Foo([1, 2, 3])
assert_type(obj.where_greater_zero(), NDArray)
