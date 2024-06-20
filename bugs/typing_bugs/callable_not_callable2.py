#!/usr/bin/env python

from typing import Callable, Protocol, TypeAlias, TypeVar

T = TypeVar("T")
S = TypeVar("S")


class SupportsGetItem(Protocol[S, T]):
    def __getitem__(self, index: S) -> T: ...


MaybeWrapped: TypeAlias = T | SupportsGetItem[int, T] | Callable[[int], T]


class Foo: ...


class Bar: ...


O = TypeVar("O", Foo, Bar)


def get_initialization(x: MaybeWrapped[O], k: int, /) -> Foo:
    match x:
        case Callable() as func:
            return func(k)

        case SupportsGetItem() as box:
            return box[k]
        case _:
            return x
