r"""Protocol variants of common ABCs."""
# ruff: noqa: N805

__all__ = [
    "BaseBuffer",
    "ReadBuffer",
    "WriteBuffer",
    "Buffer",
    "Vec",
    "Seq",
    "MutSeq",
    "Map",
    "MutMap",
    "Set",
]


from abc import abstractmethod
from collections.abc import (
    Collection,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    ValuesView,
)
from typing import (
    Optional,
    Protocol,
    Self,
    TypeIs,
    _ProtocolMeta as ProtocolMeta,
    cast,
    overload,
    runtime_checkable,
)

from ._protocols import SupportsKeysAndGetItem


@runtime_checkable
class BaseBuffer(Protocol):
    r"""Base class for ReadBuffer and WriteBuffer."""

    # SEE: WriteBuffer from https://github.com/pandas-dev/pandas/blob/main/pandas/_typing.py
    # SEE: SupportsWrite from https://github.com/python/typeshed/blob/main/stdlib/_typeshed/__init__.pyi
    # SEE: IOBase from https://github.com/python/typeshed/blob/main/stdlib/io.pyi
    # SEE: IO from https://github.com/python/typeshed/blob/main/stdlib/typing.pyi
    @property
    def mode(self) -> str: ...
    def seek(self, offset: int, whence: int = ..., /) -> int: ...
    def seekable(self) -> bool: ...
    def tell(self) -> int: ...


@runtime_checkable
class ReadBuffer[io: (str, bytes)](BaseBuffer, Protocol):
    r"""Protocol for objects that support reading."""

    def read(self, size: int = ..., /) -> io: ...


@runtime_checkable
class WriteBuffer[io: (str, bytes)](BaseBuffer, Protocol):
    r"""Protocol for objects that support writing."""

    def write(self, content: io, /) -> object: ...
    def flush(self) -> object: ...


@runtime_checkable
class Buffer[io: (str, bytes)](ReadBuffer[io], WriteBuffer[io], Protocol):
    r"""Protocol for objects that support reading and writing."""


class _VecMeta(ProtocolMeta):
    def __subclasscheck__(cls, other: type, /) -> TypeIs[type[Vec]]:
        if issubclass(other, str | bytes | Mapping):
            return False
        return super().__subclasscheck__(other)


@runtime_checkable
class Vec[T](Protocol, metaclass=_VecMeta):  # +T
    r"""Alternative to `Sequence` without `__reversed__`, `index` and `count`.

    We remove these methods, as they are not present on certain vector data structures,
    for example, `__reversed__` is not present on `pandas.Index`.

    Note:
        This class uses special casing code that ensures `str`, `bytes` and `Mapping`
        types are not considered as subtypes. Any class considered a subclass of `Mapping`
        will fail `issubclass`, in particular also `dict`.

    Examples:
        - list
        - tuple
        - numpy.ndarray
        - pandas.Index
        - pandas.Series (with integer index)
    Counter-Example:
        - `str`/`bytes` (__contains__ incompatible)
        - `dict`
    """

    @abstractmethod
    def __len__(self) -> int: ...

    # Mixin methods
    def __iter__(self) -> Iterator[T]:
        for i in range(len(self)):
            yield self[i]

    def __contains__(self, value: object, /) -> bool:
        return any(x == value or x is value for x in self)

    @overload
    @abstractmethod
    def __getitem__(self, index: int, /) -> T: ...
    @overload
    @abstractmethod
    # NOTE: not "-> Self" to ensure compatibility with tuple.
    def __getitem__(self, index: slice, /) -> Vec[T]: ...


@runtime_checkable
class Seq[T](Protocol):  # +T
    r"""Protocol version of `collections.abc.Sequence`.

    Note:
        Only compatible with `tuple[T, ...]`, not `tuple[*Ts]` when using pyright.

    References:
        - https://github.com/python/typeshed/blob/main/stdlib/typing.pyi
        - https://github.com/python/cpython/blob/main/Lib/_collections_abc.py
    """

    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __iter__(self) -> Iterator[T]: ...
    @abstractmethod
    def __contains__(self, value: object, /) -> bool: ...
    @overload
    @abstractmethod
    def __getitem__(self, index: int, /) -> T: ...
    @overload
    @abstractmethod
    def __getitem__(self, index: slice, /) -> Self: ...


@runtime_checkable
class MutSeq[T](Seq[T], Protocol):
    r"""Protocol version of `collections.abc.MutableSequence`."""

    @overload
    def __setitem__(self, index: int, value: T, /) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[T], /) -> None: ...
    @overload
    def __delitem__(self, index: int, /) -> None: ...
    @overload
    def __delitem__(self, index: slice, /) -> None: ...

    # Mixin Methods
    def __iadd__(self, values: Iterable[T], /) -> Self:
        self.extend(values)
        return self

    def insert(self, index: int, value: T, /) -> None: ...

    def append(self, value: T, /) -> None:
        self.insert(len(self), value)

    def extend(self, values: Iterable[T], /) -> None:
        for idx, value in enumerate(values, start=len(self)):
            self.insert(idx, value)

    def pop(self, index: int = -1, /) -> T:
        value = self[index]
        del self[index]
        return value

    def clear(self, /) -> None:
        del self[:]


@runtime_checkable
class Map[K, V](Collection[K], Protocol):  # K, +V
    r"""Protocol version of `collections.abc.Mapping`."""

    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, key: K, /) -> V: ...

    # Mixin Methods
    def keys(self) -> KeysView[K]:
        # NOTE: MappingView really only needs __contains__, __iter__, and __getitem__.
        return KeysView(cast("Mapping", self))

    def values(self) -> ValuesView[V]:
        return ValuesView(cast("Mapping", self))

    def items(self) -> ItemsView[K, V]:
        return ItemsView(cast("Mapping", self))

    # NOTE: dict.get has default as positional-only, whereas Mapping defines it as
    #   positional-or-keyword. We follow the weaker dict definition.
    @overload
    def get(self, key: K, /) -> Optional[V]: ...
    @overload
    def get[T](self, key: K, default: V | T, /) -> V | T: ...
    def get[T](self, key: K, default: Optional[V | T] = None, /) -> Optional[V | T]:
        try:
            return self[key]
        except KeyError:
            return default

    def __eq__(self, other: object, /) -> bool:
        if not isinstance(other, Map):
            return NotImplemented
        return dict(self.items()) == dict(other.items())

    def __contains__(self, key: object, /) -> bool:
        try:
            self[key]  # type: ignore
        except KeyError:
            return False
        return True


@runtime_checkable
class MutMap[K, V](Map[K, V], Protocol):
    r"""Protocol version of `collections.abc.MutableMapping`."""

    @abstractmethod
    def __setitem__(self, key: K, value: V, /) -> None: ...
    @abstractmethod
    def __delitem__(self, key: K, /) -> None: ...

    # FIXME: implement mixin methods
    def clear(self) -> None: ...

    # NOTE: dict.pop has default as positional-only, whereas Mapping defines it as
    #   positional-or-keyword. We follow the weaker dict definition.
    # fmt: off
    @overload
    def pop(self, key: K, /) -> V: ...
    @overload
    def pop(self, key: K, default: V, /) -> V: ...
    @overload
    def pop[T](self, key: K, default: T, /) -> V | T: ...
    def popitem(self) -> tuple[K, V]: ...
    @overload
    def setdefault[T](self: MutMap[K, T | None], key: K, default: None = ..., /) -> T | None: ...
    @overload
    def setdefault(self, key: K, default: V, /) -> V: ...
    @overload
    def update(self, m: SupportsKeysAndGetItem[K, V], /, **kwargs: V) -> None: ...
    @overload
    def update(self, m: Iterable[tuple[K, V]], /, **kwargs: V) -> None: ...
    @overload
    def update(self, **kwargs: V) -> None: ...
    # fmt: on


@runtime_checkable
class Set[V](Protocol):  # +V
    r"""Protocol version of `collections.abc.Set`."""

    # abstract methods
    def __contains__(self, value: object, /) -> bool: ...
    def __iter__(self) -> Iterator[V]: ...
    def __len__(self) -> int: ...

    # mixin methods
    # set arithmetic
    def __and__(self, other: Set, /) -> Self: ...
    def __or__[T](self, other: Set[T], /) -> Set[T | V]: ...
    def __sub__(self, other: Set, /) -> Self: ...
    def __xor__[T](self, other: Set[T], /) -> Set[T | V]: ...

    # set comparison
    def __le__(self, other: Set, /) -> bool: ...
    def __lt__(self, other: Set, /) -> bool: ...
    def __ge__(self, other: Set, /) -> bool: ...
    def __gt__(self, other: Set, /) -> bool: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __ne__(self, other: object, /) -> bool: ...
    def isdisjoint(self, other: Iterable, /) -> bool: ...
