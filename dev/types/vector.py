#!/usr/bin/env python

from typing import Protocol, Self, TypeVar

ScalarType = TypeVar("ScalarType")


class Vector(Protocol[ScalarType]):
    """An element of an abstract vector space."""

    def __len__(self) -> int: ...
    def __pos__(self) -> Self: ...
    def __neg__(self) -> Self: ...
    def __add__(self, other: Self, /) -> Self: ...
    def __sub__(self, other: Self, /) -> Self: ...
    def __mul__(self, other: ScalarType, /) -> Self: ...
    def __getitem__(self, key: int, /) -> ScalarType: ...


class MyVector(Vector[ScalarType]):
    def __init__(self, data: list[ScalarType]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __pos__(self) -> Self:
        return self

    def __neg__(self) -> Self:
        return MyVector([-x for x in self.data])

    def __add__(self, other: Self, /) -> Self:
        return MyVector([x + y for x, y in zip(self.data, other.data)])

    def __sub__(self, other: Self, /) -> Self:
        return MyVector([x - y for x, y in zip(self.data, other.data)])

    def __mul__(self, other: ScalarType, /) -> Self:
        return MyVector([x * other for x in self.data])

    def __getitem__(self, key: int, /) -> ScalarType:
        return self.data[key]


v: Vector[float] = MyVector([1.0, 2.0, 3.0])


def scalar_multiplication(vector, scalar):
    """Multiply a vector by a compatible scalar."""
    return scalar * vector


import numpy as np

v: Vector[np.float64] = np.array([1.0, 2.0, 3.0])

x * 17
