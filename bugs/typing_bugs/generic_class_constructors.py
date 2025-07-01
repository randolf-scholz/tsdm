# fmt: off
from typing import Any, Self, assert_type


def test_return_self() -> None:
    r"""from_list(cls, list[T]) -> Self."""
    class Array[T = Any]:
        @classmethod
        def from_list(cls, x: list[T], /) -> Self:
            return cls()

    class IntArray(Array[int]): ...
    class SubArray[T=Any](Array[T]): ...

    assert_type(Array.from_list([])        , Array[Any])
    assert_type(Array.from_list([1])       , Array[int])
    assert_type(Array[int].from_list([])   , Array[int])
    assert_type(Array[int].from_list([1])  , Array[int])
    assert_type(Array[str].from_list([])   , Array[str])
    assert_type(Array[str].from_list(["a"]), Array[str])

    assert_type(SubArray.from_list([])        , SubArray[Any])
    assert_type(SubArray.from_list([1])       , SubArray[int])
    assert_type(SubArray[int].from_list([])   , SubArray[int])
    assert_type(SubArray[int].from_list([1])  , SubArray[int])
    assert_type(SubArray[str].from_list([])   , SubArray[str])
    assert_type(SubArray[str].from_list(["a"]), SubArray[str])

    assert_type(IntArray.from_list([])   , IntArray)
    assert_type(IntArray.from_list([1])  , IntArray)


def test_bind_using_class_typevar() -> None:
    r"""from_list(type[Array[T]], list[T]) -> Array[T]."""

    class Array[T = Any]:
        @classmethod
        def from_list(cls: "type[Array[T]]", x: list[T], /) -> "Array[T]":
            return cls()

    class IntArray(Array[int]): ...
    class SubArray[T](Array[T]): ...

    assert_type(Array.from_list([])        , Array[Any])
    assert_type(Array.from_list([1])       , Array[int])
    assert_type(Array[int].from_list([])   , Array[int])
    assert_type(Array[int].from_list([1])  , Array[int])
    assert_type(Array[str].from_list([])   , Array[str])
    assert_type(Array[str].from_list(["a"]), Array[str])

    assert_type(SubArray.from_list([])        , SubArray[Any])
    assert_type(SubArray.from_list([1])       , SubArray[int])
    assert_type(SubArray[int].from_list([])   , SubArray[int])
    assert_type(SubArray[int].from_list([1])  , SubArray[int])
    assert_type(SubArray[str].from_list([])   , SubArray[str])
    assert_type(SubArray[str].from_list(["a"]), SubArray[str])

    assert_type(IntArray.from_list([])   , IntArray)
    assert_type(IntArray.from_list([1])  , IntArray)

def test_bind_using_method_typevar() -> None:
    r"""from_list[X](type[Array[X]], list[X]) -> Array[X]."""
    class Array[T = Any]:
        @classmethod
        def from_list[X = Any](cls: "type[Array[X]]", x: list[X], /) -> "Array[X]":
            return cls()

    class IntArray(Array[int]): ...
    class SubArray[T](Array[T]): ...

    assert_type(Array.from_list([])        , Array[Any])
    assert_type(Array.from_list([1])       , Array[int])
    assert_type(Array[int].from_list([])   , Array[int])
    assert_type(Array[int].from_list([1])  , Array[int])
    assert_type(Array[str].from_list([])   , Array[str])
    assert_type(Array[str].from_list(["a"]), Array[str])

    assert_type(SubArray.from_list([])        , SubArray[Any])
    assert_type(SubArray.from_list([1])       , SubArray[int])
    assert_type(SubArray[int].from_list([])   , SubArray[int])
    assert_type(SubArray[int].from_list([1])  , SubArray[int])
    assert_type(SubArray[str].from_list([])   , SubArray[str])
    assert_type(SubArray[str].from_list(["a"]), SubArray[str])

    assert_type(IntArray.from_list([])   , IntArray)
    assert_type(IntArray.from_list([1])  , IntArray)


def test_bind_using_metaclass_naive() -> None:
    r"""Meta.from_list[T](cls, list[T]) -> Array[T]."""

    class Meta(type):
        def from_list[T = Any](cls, x: list[T], /) -> Self:
            return cls()

    class Array[X = Any](metaclass=Meta): ...
    class IntArray(Array[int]): ...
    class SubArray[T](Array[T]): ...

    assert_type(Array.from_list([])        , Array[Any])
    assert_type(Array.from_list([1])       , Array[int])
    assert_type(Array[int].from_list([])   , Array[int])
    assert_type(Array[int].from_list([1])  , Array[int])
    assert_type(Array[str].from_list([])   , Array[str])
    assert_type(Array[str].from_list(["a"]), Array[str])

    assert_type(SubArray.from_list([])        , SubArray[Any])
    assert_type(SubArray.from_list([1])       , SubArray[int])
    assert_type(SubArray[int].from_list([])   , SubArray[int])
    assert_type(SubArray[int].from_list([1])  , SubArray[int])
    assert_type(SubArray[str].from_list([])   , SubArray[str])
    assert_type(SubArray[str].from_list(["a"]), SubArray[str])

    assert_type(IntArray.from_list([])   , IntArray)
    assert_type(IntArray.from_list([1])  , IntArray)


def test_bind_using_metaclass() -> None:
    r"""Meta.from_list[T](cls, list[T]) -> Array[T]."""

    class Meta(type):
        def from_list[T = Any](cls, x: list[T], /) -> "Array[T]":
            return cls()

    class Array[X = Any](metaclass=Meta): ...
    class IntArray(Array[int]): ...
    class SubArray[T](Array[T]): ...

    assert_type(Array.from_list([])        , Array[Any])
    assert_type(Array.from_list([1])       , Array[int])
    assert_type(Array[int].from_list([])   , Array[int])
    assert_type(Array[int].from_list([1])  , Array[int])
    assert_type(Array[str].from_list([])   , Array[str])
    assert_type(Array[str].from_list(["a"]), Array[str])

    assert_type(SubArray.from_list([])        , SubArray[Any])
    assert_type(SubArray.from_list([1])       , SubArray[int])
    assert_type(SubArray[int].from_list([])   , SubArray[int])
    assert_type(SubArray[int].from_list([1])  , SubArray[int])
    assert_type(SubArray[str].from_list([])   , SubArray[str])
    assert_type(SubArray[str].from_list(["a"]), SubArray[str])

    assert_type(IntArray.from_list([])   , IntArray)
    assert_type(IntArray.from_list([1])  , IntArray)

def test_bind_using_metaclass_with_typevar() -> None:
    r"""Meta.from_list[T](cls: type[Array[T]], list[T]) -> Array[T]."""

    class Meta(type):
        def from_list[T = Any](cls: "type[Array[T]]", x: list[T], /) -> "Array[T]":
            return cls()

    class Array[X = Any](metaclass=Meta): ...
    class IntArray(Array[int]): ...
    class SubArray[T](Array[T]): ...

    assert_type(Array.from_list([])        , Array[Any])
    assert_type(Array.from_list([1])       , Array[int])
    assert_type(Array[int].from_list([])   , Array[int])
    assert_type(Array[int].from_list([1])  , Array[int])
    assert_type(Array[str].from_list([])   , Array[str])
    assert_type(Array[str].from_list(["a"]), Array[str])

    assert_type(SubArray.from_list([])        , SubArray[Any])
    assert_type(SubArray.from_list([1])       , SubArray[int])
    assert_type(SubArray[int].from_list([])   , SubArray[int])
    assert_type(SubArray[int].from_list([1])  , SubArray[int])
    assert_type(SubArray[str].from_list([])   , SubArray[str])
    assert_type(SubArray[str].from_list(["a"]), SubArray[str])

    assert_type(IntArray.from_list([])   , IntArray)
    assert_type(IntArray.from_list([1])  , IntArray)


def test_bind_using_metaclass_with_typevar_error() -> None:
    r"""Meta.from_list[T = Any](cls: type[Array[T]], list[T]) -> Array[T]."""

    class Meta(type):
        def from_list[T = Any](cls: "type[Array[T]]", x: list[T], /) -> "Array[T]":  # error in mypy and pyright
            return cls()

    class Array[X = Any](metaclass=Meta): ...
    class IntArray(Array[int]): ...
    class SubArray[T](Array[T]): ...

    assert_type(Array.from_list([])        , Array[Any])
    assert_type(Array.from_list([1])       , Array[int])
    assert_type(Array[int].from_list([])   , Array[int])
    assert_type(Array[int].from_list([1])  , Array[int])
    assert_type(Array[str].from_list([])   , Array[str])
    assert_type(Array[str].from_list(["a"]), Array[str])

    assert_type(SubArray.from_list([])        , SubArray[Any])
    assert_type(SubArray.from_list([1])       , SubArray[int])
    assert_type(SubArray[int].from_list([])   , SubArray[int])
    assert_type(SubArray[int].from_list([1])  , SubArray[int])
    assert_type(SubArray[str].from_list([])   , SubArray[str])
    assert_type(SubArray[str].from_list(["a"]), SubArray[str])

    assert_type(IntArray.from_list([])   , IntArray)
    assert_type(IntArray.from_list([1])  , IntArray)


def test_with_metaclass_bind_array() -> None:
    r"""Meta.from_list[Arr: Array](type[A], list) -> A."""
    class Meta(type):
        def from_list[A: "Array"](cls: "type[A]", x: list, /) -> "A":
            return cls()

    class Array[X = Any](metaclass=Meta): ...
    class IntArray(Array[int]): ...
    class SubArray[T](Array[T]): ...

    assert_type(Array.from_list([])        , Array[Any])
    assert_type(Array.from_list([1])       , Array[int])
    assert_type(Array[int].from_list([])   , Array[int])
    assert_type(Array[int].from_list([1])  , Array[int])
    assert_type(Array[str].from_list([])   , Array[str])
    assert_type(Array[str].from_list(["a"]), Array[str])

    assert_type(SubArray.from_list([])        , SubArray[Any])
    assert_type(SubArray.from_list([1])       , SubArray[int])
    assert_type(SubArray[int].from_list([])   , SubArray[int])
    assert_type(SubArray[int].from_list([1])  , SubArray[int])
    assert_type(SubArray[str].from_list([])   , SubArray[str])
    assert_type(SubArray[str].from_list(["a"]), SubArray[str])

    assert_type(IntArray.from_list([])   , IntArray)
    assert_type(IntArray.from_list([1])  , IntArray)

def test_with_metaclass_bind_array_generic() -> None:
    r"""Meta.from_list[X, A: Array](type[A], list[X]) -> A."""
    class Meta(type):
        def from_list[X=Any, A: "Array[X]"](cls: "type[A]", x: list[X], /) -> "A":
            return cls()

    class Array[Z = Any](metaclass=Meta): ...
    class IntArray(Array[int]): ...
    class SubArray[Z=Any](Array[Z]): ...

    assert_type(Array.from_list([])        , Array[Any])
    assert_type(Array.from_list([1])       , Array[int])
    assert_type(Array[int].from_list([])   , Array[int])
    assert_type(Array[int].from_list([1])  , Array[int])
    assert_type(Array[str].from_list([])   , Array[str])
    assert_type(Array[str].from_list(["a"]), Array[str])

    assert_type(SubArray.from_list([])        , SubArray[Any])
    assert_type(SubArray.from_list([1])       , SubArray[int])
    assert_type(SubArray[int].from_list([])   , SubArray[int])
    assert_type(SubArray[int].from_list([1])  , SubArray[int])
    assert_type(SubArray[str].from_list([])   , SubArray[str])
    assert_type(SubArray[str].from_list(["a"]), SubArray[str])

    assert_type(IntArray.from_list([])   , IntArray)
    assert_type(IntArray.from_list([1])  , IntArray)



def test_demo() -> None:
    class Array[T = Any]: ...
    class IntArray(Array[int]): ...
    class SubArray[T=Any](Array[T]): ...

    assert_type(Array.from_list([])        , Array[Any])
    assert_type(Array.from_list([1])       , Array[int])
    assert_type(Array[int].from_list([])   , Array[int])
    assert_type(Array[int].from_list([1])  , Array[int])
    assert_type(Array[str].from_list([])   , Array[str])
    assert_type(Array[str].from_list(["a"]), Array[str])

    assert_type(SubArray.from_list([])        , SubArray[Any])
    assert_type(SubArray.from_list([1])       , SubArray[int])
    assert_type(SubArray[int].from_list([])   , SubArray[int])
    assert_type(SubArray[int].from_list([1])  , SubArray[int])
    assert_type(SubArray[str].from_list([])   , SubArray[str])
    assert_type(SubArray[str].from_list(["a"]), SubArray[str])

    assert_type(IntArray.from_list([])   , IntArray)
    assert_type(IntArray.from_list([1])  , IntArray)
