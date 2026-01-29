r"""Small mixin protocols."""

__all__ = [
    "Orderable",
    "SupportsBool",
    "SupportsGetItem",
    "SupportsKeysAndGetItem",
    "SupportsLenAndGetItem",
    "SupportsSlicing",
]


from abc import abstractmethod
from collections.abc import Collection
from typing import Any, Protocol, Self, overload, runtime_checkable


@runtime_checkable
class Orderable(Protocol):
    r"""Protocol for types that support ordering operations."""

    def __ge__(self, other: Any, /) -> object: ...
    def __gt__(self, other: Any, /) -> object: ...
    def __le__(self, other: Any, /) -> object: ...
    def __lt__(self, other: Any, /) -> object: ...


@runtime_checkable
class SupportsBool(Protocol):
    r"""Protocol for types that support boolean operations."""

    def __bool__(self) -> bool: ...


@runtime_checkable
class SupportsGetItem[K, V](Protocol):  # -K, +V
    r"""Protocol for objects that support `__getitem__`."""

    @abstractmethod
    def __getitem__(self, key: K, /) -> V: ...


@runtime_checkable
class SupportsKeysAndGetItem[K, V](Protocol):  # K, +V
    r"""Protocol for objects that support `__getitem__` and `keys`."""

    @abstractmethod
    def keys(self) -> Collection[K]: ...
    @abstractmethod
    def __getitem__(self, key: K, /) -> V: ...


@runtime_checkable
class SupportsLenAndGetItem[V](Protocol):  # +V
    r"""Protocol for objects that support integer based `__getitem__` and `__len__`."""

    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __getitem__(self, index: int, /) -> V: ...


@runtime_checkable
class SupportsSlicing[V](SupportsGetItem[int, V], Protocol):  # +V
    r"""Protocol for objects that support slicing with integer indices."""

    @overload
    @abstractmethod
    def __getitem__(self, index: int, /) -> V: ...
    @overload
    @abstractmethod
    def __getitem__(self, index: slice, /) -> Self: ...
