# fmt: off
from typing import Any, Self, assert_type


class Array[T = Any]:
    @classmethod
    def from_list(cls, x: list[T], /) -> Self:
        return cls()

assert_type(Array.from_list([])      , Array[Any])
assert_type(Array.from_list([1])     , Array[int])
assert_type(Array[int].from_list([]) , Array[int])
assert_type(Array[int].from_list([1]), Array[int])

class Array1[T = Any]:
    @classmethod
    def from_list(cls, x: list[T], /) -> Self:
        return cls()

assert_type(Array1.from_list([])      , Array1[Any])
assert_type(Array1.from_list([1])     , Array1[int])  # Error in pyright
assert_type(Array1[int].from_list([]) , Array1[int])
assert_type(Array1[int].from_list([1]), Array1[int])

class Array2[T = Any]:
    @classmethod
    def from_list(cls: "type[Array2[T]]", x: list[T], /) -> "Array2[T]":
        return cls()

assert_type(Array2.from_list([])      , Array2[Any])
assert_type(Array2.from_list([1])     , Array2[int])  # Error in pyright
assert_type(Array2[int].from_list([]) , Array2[int])
assert_type(Array2[int].from_list([1]), Array2[int])

class Array3[T = Any]:
    @classmethod
    def from_list[X = Any](cls: "type[Array3[X]]", x: list[X], /) -> "Array3[X]":
        return cls()

assert_type(Array3.from_list([])      , Array3[Any])
assert_type(Array3.from_list([1])     , Array3[int])  # Error in pyright
assert_type(Array3[int].from_list([]) , Array3[int])
assert_type(Array3[int].from_list([1]), Array3[int])

class Array4Meta(type):
    def from_list[T = Any](cls, x: list[T], /) -> "Array4[T]":
        return cls()

class Array4[T](metaclass=Array4Meta): ...

assert_type(Array4.from_list([])      , Array4[Any])
assert_type(Array4.from_list([1])     , Array4[int])
assert_type(Array4[int].from_list([]) , Array4[int])  # Error in mypy and pyright
assert_type(Array4[int].from_list([1]), Array4[int])

class Array5Meta(type):
    def from_list[T = Any](cls: "type[Array5[T]]", x: list[T], /) -> "Array5[T]":  # error in mypy and pyright
        return cls()

class Array5[T = Any](metaclass=Array5Meta): ...

assert_type(Array5.from_list([])      , Array5[Any])  # Error in mypy
assert_type(Array5.from_list([1])     , Array5[int])  # Error in mypy and pyright
assert_type(Array5[int].from_list([]) , Array5[int])
assert_type(Array5[int].from_list([1]), Array5[int])
