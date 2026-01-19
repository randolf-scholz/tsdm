from typing import Literal, Mapping


class Base[Key]:
    items: Mapping[Key, int]


class ImplA(Base[Literal["foo", "bar"]]):
    items: Mapping[Literal["foo", "bar"], int] = {"foo": 1, "bar": 2}  # ✅️


class ImplB(Base[Literal["foo", "bar"]]):
    items = {"foo": 1, "bar": 2}  # ❌️
