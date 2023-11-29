#!/usr/bin/env python

from types import GenericAlias
from typing import Generic, Iterator, Protocol, TypeVar, runtime_checkable

T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class GenericIterable(Protocol[T_co]):
    def __class_getitem__(cls, item: type) -> GenericAlias: ...
    def __iter__(self) -> Iterator[T_co]: ...


x: GenericIterable[str] = ["a", "b", "c"]
y: GenericIterable[str] = "abc"
