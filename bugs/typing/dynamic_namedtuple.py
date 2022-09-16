#!/usr/bin/env python

from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from collections import namedtuple
from collections.abc import Iterable
from typing import NamedTuple, Protocol, TypeGuard, TypeVar, cast

T = TypeVar("T")


class _NamedTuple(tuple[T, ...], ABC):
    r"""To check for namedtuple."""
    __slots__ = ()

    @abstractmethod
    def __init__(self, *args: T) -> None:
        ...

    @classmethod
    @abstractmethod
    def _make(cls, iterable: Iterable[T]) -> _NamedTuple[T]:
        ...

    @abstractmethod
    def _replace(self, /, **kwds: dict[str, T]) -> None:
        ...

    @property
    @abstractmethod
    def _fields(self) -> tuple[T, ...]:
        ...

    @property
    @abstractmethod
    def _field_defaults(self) -> dict[str, T]:
        ...

    @abstractmethod
    def _asdict(self) -> dict[str, T]:
        ...


def register_namedtuple(fields: list[str], /, *, name: str) -> type[_NamedTuple]:
    if not name.isidentifier():
        raise ValueError(f"{name} is not a valid identifier!")
    tuple_type: type[tuple] = namedtuple(name, fields)  # type: ignore[misc]
    reveal_type(tuple_type)
    _NamedTuple.register(tuple_type)
    return cast(type[_NamedTuple], tuple_type)


class Foo:
    tuple: type[_NamedTuple]

    def __init__(self, fields: list[str]):
        super().__init__()
        self.tuple = register_namedtuple(fields, name="FooTuple")


foo = Foo(["a", "b", "c"])
FooTup = foo.tuple
footup = foo.tuple(1, 2, 3)

assert isinstance(footup, tuple)
assert issubclass(FooTup, tuple)
assert isinstance(footup, _NamedTuple)
assert issubclass(FooTup, _NamedTuple)
