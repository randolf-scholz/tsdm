#!/usr/bin/env python

from typing import Generic, Literal, TypeVar

# class A(Generic[K]):
#     table_names: list[K]
#
#
# class B(A[KEYS]):
#     table_names = ["foo", "bar"]  # ❌
#     # Expression of type "list[str]" cannot be assigned to declared type "list[KEYS]"

#
# class C(A[KEYS]): ...
#
#
# reveal_type(C.table_names)  # "C.table_names" is "list[Literal['foo', 'bar']]"
# C.table_names = ["foo", "bar"]  # ✅
#


K = TypeVar("K")


class A(Generic[K]):
    table_names: list[K]


class B(A[int]):
    table_names = [
        "1",
        "2",
    ]  # "list[str]" cannot be assigned to declared type "list[int]" ✅


class C(A[int]):
    table_names = [1, 2]  # ✅
