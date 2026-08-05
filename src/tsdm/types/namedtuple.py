r"""NamedTuple Protocol and utilities."""

__all__ = [
    "NTuple",
    "is_namedtuple",
    "issubclass_namedtuple",
    "isinstance_namedtuple",
]


import typing
from abc import ABCMeta
from collections.abc import Iterator, Mapping
from types import get_original_bases
from typing import (
    ClassVar,
    Final,
    Protocol,
    SupportsIndex,
    TypeIs,
    _ProtocolMeta as ProtocolMeta,
    overload,
    runtime_checkable,
)

import typing_extensions


class _NTupleMeta(ProtocolMeta):
    r"""Metaclass for `NTuple`.

    Note:
        We use this as an alternative to defining `__subclasshook__`
        See https://github.com/python/cpython/issues/106363
    """

    _fields: ClassVar[tuple[str, ...]] = ()
    r"""The fields of the namedtuple."""

    def __instancecheck__(cls, instance: object, /) -> TypeIs[NTuple]:  # noqa: N805
        return cls.__subclasscheck__(type(instance))

    def __subclasscheck__(cls, subclass: type, /) -> TypeIs[type[NTuple]]:  # noqa: N805
        if ABCMeta.__subclasscheck__(cls, subclass):
            return True
        bases = get_original_bases(subclass)
        return (typing.NamedTuple in bases) or (typing_extensions.NamedTuple in bases)


@runtime_checkable  # FIXME: Use TypeVarTuple
class NTuple[T](Protocol, metaclass=_NTupleMeta):  # +T
    r"""Protocol for anonymous namedtuple.

    References:
        - https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi
    """

    # FIXME: Problems if denoted as  tuple[str, ...]
    #    Should be tuple[*(str for T in Ts)] (not tuple[str, ...])
    #   see: https://github.com/python/typing/issues/1216
    #   see: https://github.com/python/typing/issues/1273

    _fields: Final[ClassVar[tuple[str, ...]]]  # type: ignore
    r"""The fields of the namedtuple."""

    def _asdict(self) -> Mapping[str, T]: ...
    def __contains__(self, key: object, /) -> bool: ...
    def __iter__(self) -> Iterator[T]: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: SupportsIndex, /) -> T: ...
    @overload
    def __getitem__(self, key: slice, /) -> tuple[T, ...]: ...

    # def __lt__(self, value: tuple[T, ...], /) -> bool: ...
    # def __le__(self, value: tuple[T, ...], /) -> bool: ...
    # def __gt__(self, value: tuple[T, ...], /) -> bool: ...
    # def __ge__(self, value: tuple[T, ...], /) -> bool: ...
    # @overload
    # def __add__(self, value: tuple[T, ...], /) -> tuple[T, ...]: ...
    # @overload
    # def __add__[T2](self, value: tuple[T2, ...], /) -> tuple[T | T2, ...]: ...
    # def __mul__(self, value: SupportsIndex, /) -> tuple[T, ...]: ...
    # def __rmul__(self, value: SupportsIndex, /) -> tuple[T, ...]: ...
    # def count(self, value: Any, /) -> int: ...
    # def index(self, value: Any, start: SupportsIndex = 0, stop: SupportsIndex = sys.maxsize, /) -> int: ...


def issubclass_namedtuple(cls: type, /) -> TypeIs[type[NTuple]]:
    return issubclass(cls, NTuple)  # type: ignore


def isinstance_namedtuple(obj: object, /) -> TypeIs[NTuple]:
    return issubclass(type(obj), NTuple)  # type: ignore


@overload
def is_namedtuple(obj: type, /) -> TypeIs[type[NTuple]]: ...
@overload
def is_namedtuple(obj: object, /) -> TypeIs[NTuple]: ...
def is_namedtuple(obj: object, /) -> TypeIs[NTuple] | TypeIs[type[NTuple]]:
    r"""Check if the object is a namedtuple."""
    if isinstance(obj, type):
        return issubclass(obj, NTuple)  # type: ignore
    return issubclass(type(obj), NTuple)  # type: ignore
