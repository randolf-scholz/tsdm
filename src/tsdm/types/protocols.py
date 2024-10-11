r"""Protocols used in tsdm.

References:
    - https://www.python.org/dev/peps/pep-0544/
    - https://docs.python.org/3/library/typing.html#typing.Protocol
    - https://numpy.org/doc/stable/reference/c-api/array.html
"""

__all__ = [
    # Classes
    "Hash",
    "Lookup",
    "ShapeLike",
    "Array",
    # stdlib
    "Map",
    "MutMap",
    "MutSeq",
    "Seq",
    "SetProtocol",
    "SupportsGetItem",
    "SupportsKeysAndGetItem",
    "SupportsKwargs",
    "SupportsLenAndGetItem",
    # other
    "BaseBuffer",
    "Buffer",
    "ReadBuffer",
    "WriteBuffer",
    "GenericIterable",
    "_SupportsKwargsMeta",
    # Factory classes
    "Dataclass",
    "NTuple",
    "Slotted",
    # Functions
    "is_dataclass",
    "isinstance_dataclass",
    "issubclass_dataclass",
    "isinstance_namedtuple",
    "issubclass_namedtuple",
    "is_namedtuple",
    "is_slotted",
]

import dataclasses
import typing
from abc import ABCMeta, abstractmethod
from collections.abc import (
    Collection,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    ValuesView,
)
from types import GenericAlias, get_original_bases
from typing import (
    Any,
    ClassVar,
    Final,
    Optional,
    Protocol,
    Self,
    SupportsIndex,
    _ProtocolMeta as ProtocolMeta,
    overload,
    runtime_checkable,
)

import typing_extensions
from typing_extensions import TypeIs

# region io protocols ------------------------------------------------------------------


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


# endregion io protocols ---------------------------------------------------------------


# region misc protocols ----------------------------------------------------------------
@runtime_checkable
class GenericIterable[T](Protocol):  # +T
    r"""Does not work currently!"""

    # FIXME: https://github.com/python/cpython/issues/112319
    def __class_getitem__(cls, item: type) -> GenericAlias: ...
    def __iter__(self) -> Iterator[T]: ...


@runtime_checkable
class Lookup[K, V](Protocol):  # -K, +V
    r"""Mapping/Sequence like generic that is contravariant in Keys."""

    @abstractmethod
    def __contains__(self, key: K, /) -> bool:
        # Here, any Hashable input is accepted.
        r"""Return True if the map contains the given key."""
        ...

    @abstractmethod
    def __getitem__(self, key: K, /) -> V:
        r"""Return the value associated with the given key."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        r"""Return the number of items in the map."""
        ...


@runtime_checkable
class Hash(Protocol):
    r"""Protocol for hash-functions."""

    name: str

    def digest_size(self) -> int:
        r"""Return the size of the hash in bytes."""
        ...

    def block_size(self) -> int:
        r"""Return the internal block size of the hash in bytes."""
        ...

    def update(self, data: bytes) -> None:
        r"""Update this hash object's state with the provided string."""
        ...

    def digest(self) -> bytes:
        r"""Return the digest value as a string of binary data."""
        ...

    def hexdigest(self) -> str:
        r"""Return the digest value as a string of hexadecimal digits."""
        ...

    def copy(self) -> Self:
        r"""Return a clone of the hash object."""
        ...


# endregion misc protocols -------------------------------------------------------------


# region container protocols -----------------------------------------------------------


@runtime_checkable
class ShapeLike(Protocol):
    r"""Protocol for shapes, very similar to tuple, but without `__contains__`.

    Note:
        - tensorflow.TensorShape is not a tuple, but has a similar API.
        - pytorch.Size is a tuple.
        - numpy.ndarray.shape is a tuple.
        - pandas.Series.shape is a tuple.

    References:
        - https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi
    """

    # unary operations
    def __hash__(self) -> int: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[int]: ...
    def __getitem__(self, item: int, /) -> int: ...  # int <: SupportsIndex

    # binary operations
    # NOTE: Not returning Self, because that's how tuple works.
    def __eq__(self, other: object, /) -> bool: ...
    def __ne__(self, other: object, /) -> bool: ...
    def __lt__(self, other: Self | tuple, /) -> bool: ...
    def __le__(self, other: Self | tuple, /) -> bool: ...
    # arithmetic
    def __add__(self, other: Self | tuple, /) -> "ShapeLike": ...


# endregion array protocols ------------------------------------------------------------


# region stdlib protocols --------------------------------------------------------------

# NOTE: These are added here because the standard library does not provide these as Protocols...
# Reference: https://peps.python.org/pep-0544/#changes-in-the-typing-module
# Only the following are Protocols.
# - Callable
# - Awaitable
# - Iterable, Iterator
# - AsyncIterable, AsyncIterator
# - Hashable
# - Sized
# - Container
# - Collection
# - Reversible
# - ContextManager, AsyncContextManager
# - SupportsAbs (and other Supports* classes)


@runtime_checkable
class SupportsGetItem[K, V](Protocol):  # -K, +V
    r"""Protocol for objects that support `__getitem__`."""

    def __getitem__(self, key: K, /) -> V: ...


@runtime_checkable
class SupportsKeysAndGetItem[K, V](Protocol):  # K, +V
    r"""Protocol for objects that support `__getitem__` and `keys`."""

    def keys(self) -> Iterable[K]: ...
    def __getitem__(self, key: K, /) -> V: ...


@runtime_checkable
class SupportsLenAndGetItem[V](Protocol):  # +V
    r"""Protocol for objects that support integer based `__getitem__` and `__len__`."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: int, /) -> V: ...


class _SupportsKwargsMeta(ProtocolMeta):
    r"""Metaclass for `SupportsKwargs`."""

    def __instancecheck__(cls, instance: object) -> TypeIs["SupportsKwargs"]:  # noqa: N805
        return isinstance(instance, SupportsKeysAndGetItem) and all(
            isinstance(key, str)
            for key in instance.keys()  # noqa: SIM118
        )

    def __subclasscheck__(cls, subclass: type) -> TypeIs[type["SupportsKwargs"]]:  # noqa: N805
        raise NotImplementedError("Cannot check whether a class is a SupportsKwargs.")


@runtime_checkable
class SupportsKwargs[V](Protocol, metaclass=_SupportsKwargsMeta):  # +V
    r"""Protocol for objects that support `**kwargs`."""

    def keys(self) -> Iterable[str]: ...
    def __getitem__(self, key: str, /) -> V: ...


@runtime_checkable
class SetProtocol[V](Protocol):  # +V
    r"""Protocol version of `collections.abc.Set`."""

    # abstract methods
    @abstractmethod
    def __contains__(self, value: object, /) -> bool: ...
    @abstractmethod
    def __iter__(self) -> Iterator[V]: ...
    @abstractmethod
    def __len__(self) -> int: ...

    # mixin methods
    # set arithmetic
    def __and__(self, other: "SetProtocol", /) -> Self: ...
    def __or__[T](self, other: "SetProtocol[T]", /) -> "SetProtocol[T | V]": ...
    def __sub__(self, other: "SetProtocol", /) -> Self: ...
    def __xor__[T](self, other: "SetProtocol[T]", /) -> "SetProtocol[T | V]": ...

    # set comparison
    def __le__(self, other: "SetProtocol", /) -> bool: ...
    def __lt__(self, other: "SetProtocol", /) -> bool: ...
    def __ge__(self, other: "SetProtocol", /) -> bool: ...
    def __gt__(self, other: "SetProtocol", /) -> bool: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __ne__(self, other: object, /) -> bool: ...
    def isdisjoint(self, other: Iterable, /) -> bool: ...


class _ArrayMeta(ProtocolMeta):
    def __subclasscheck__(cls, other: type) -> TypeIs[type["Array"]]:  # noqa: N805
        if issubclass(other, str | bytes | Mapping):
            return False
        return super().__subclasscheck__(other)


@runtime_checkable
class Array[T](Protocol, metaclass=_ArrayMeta):  # +T
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

    @overload
    @abstractmethod
    def __getitem__(self, index: int, /) -> T: ...
    @overload
    @abstractmethod
    # NOTE: not "-> Self" to ensure compatibility with tuple.
    def __getitem__(self, index: slice, /) -> "Array[T]": ...

    # Mixin methods
    def __iter__(self) -> Iterator[T]:
        for i in range(len(self)):
            yield self[i]

    def __contains__(self, value: object, /) -> bool:
        return any(x == value or x is value for x in self)


@runtime_checkable
class Seq[T](Protocol):  # +T
    r"""Protocol version of `collections.abc.Sequence`.

    Note:
        We intentionally exclude `Reversible`, since `tuple` fakes this:
        `tuple` has no attribute `__reversed__`, rather, it uses the
        `Sequence.register(tuple)` to artificially become a nominal subtype.

    References:
        - https://github.com/python/typeshed/blob/main/stdlib/typing.pyi
        - https://github.com/python/cpython/blob/main/Lib/_collections_abc.py
    """

    @abstractmethod
    def __len__(self) -> int: ...
    @overload
    @abstractmethod
    def __getitem__(self, index: int, /) -> T: ...
    @overload
    @abstractmethod
    def __getitem__(self, index: slice, /) -> Self: ...

    # Mixin methods
    # NOTE: intentionally excluded __reversed__
    def __iter__(self) -> Iterator[T]:
        for i in range(len(self)):
            yield self[i]

    def __contains__(self, value: object, /) -> bool:
        return any(x == value or x is value for x in self)

    @overload
    def index(self, value: Any, start: int = ..., /) -> int: ...
    @overload
    def index(self, value: Any, start: int = ..., stop: int = ..., /) -> int: ...
    def index(self, value: Any, start: int = 0, stop: None | int = None, /) -> int:
        for i, x in enumerate(self[start:stop]):
            if x == value or x is value:
                return i
        raise ValueError(f"{value!r} is not in list")

    def count(self, value: Any, /) -> int:
        return sum(x == value or x is value for x in self)


@runtime_checkable
class MutSeq[T](Seq[T], Protocol):
    r"""Protocol version of `collections.abc.MutableSequence`."""

    @overload
    @abstractmethod
    def __getitem__(self, index: int, /) -> T: ...
    @overload
    @abstractmethod
    def __getitem__(self, index: slice, /) -> Self: ...
    @overload
    @abstractmethod
    def __setitem__(self, index: int, value: T, /) -> None: ...
    @overload
    @abstractmethod
    def __setitem__(self, index: slice, value: Iterable[T], /) -> None: ...
    @overload
    @abstractmethod
    def __delitem__(self, index: int, /) -> None: ...
    @overload
    @abstractmethod
    def __delitem__(self, index: slice, /) -> None: ...
    def insert(self, index: int, value: T, /) -> None: ...

    # Mixin Methods
    def __iadd__(self, values: Iterable[T], /) -> Self:
        self.extend(values)
        return self

    def append(self, value: T, /) -> None:
        self.insert(len(self), value)

    def clear(self, /) -> None:
        del self[:]

    def extend(self, values: Iterable[T], /) -> None:
        for idx, value in enumerate(values, start=len(self)):
            self.insert(idx, value)

    def reverse(self, /) -> None:
        self[:] = self[::-1]

    def pop(self, index: int = -1, /) -> T:
        value = self[index]
        del self[index]
        return value

    def remove(self, value: T, /) -> None:
        del self[self.index(value)]


@runtime_checkable
class Map[K, V](Collection[K], Protocol):  # K, +V
    r"""Protocol version of `collections.abc.Mapping`."""

    @abstractmethod
    def __getitem__(self, __key: K, /) -> V: ...

    # Mixin Methods
    def keys(self) -> KeysView[K]:
        return KeysView(self)  # type: ignore[arg-type]

    def values(self) -> ValuesView[V]:
        return ValuesView(self)  # type: ignore[arg-type]

    def items(self) -> ItemsView[K, V]:
        return ItemsView(self)  # type: ignore[arg-type]

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
            self[key]  # type: ignore[index]
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
    @overload
    def pop(self, key: K, /) -> V: ...
    @overload
    def pop(self, key: K, default: V, /) -> V: ...
    @overload
    def pop[T](self, key: K, default: T, /) -> V | T: ...
    def popitem(self) -> tuple[K, V]: ...
    @overload
    def setdefault[T](
        self: "MutMap[K, T | None]", key: K, default: None = ..., /
    ) -> T | None: ...
    @overload
    def setdefault(self, key: K, default: V, /) -> V: ...
    @overload
    def update(self, m: SupportsKeysAndGetItem[K, V], /, **kwargs: V) -> None: ...
    @overload
    def update(self, m: Iterable[tuple[K, V]], /, **kwargs: V) -> None: ...
    @overload
    def update(self, **kwargs: V) -> None: ...


# endregion stdlib protocols -----------------------------------------------------------


# region generic factory-protocols -----------------------------------------------------


class _DataclassMeta(ProtocolMeta):
    r"""Metaclass for `Dataclass`."""

    def __instancecheck__(cls, instance: object) -> TypeIs["Dataclass"]:  # noqa: N805
        return cls.__subclasscheck__(type(instance))

    def __subclasscheck__(cls, subclass: type) -> TypeIs[type["Dataclass"]]:  # noqa: N805
        fields = getattr(subclass, "__dataclass_fields__", None)
        return isinstance(fields, dict)


@runtime_checkable
class Dataclass(Protocol, metaclass=_DataclassMeta):
    r"""Protocol for anonymous dataclasses.

    Similar to `DataClassInstance` from typeshed, but allows isinstance and issubclass.
    """

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]] = {}
    r"""The fields of the dataclass."""


class _NTupleMeta(ProtocolMeta):
    r"""Metaclass for `NTuple`.

    Note:
        We use this as an alternative to defining `__subclasshook__`
        See https://github.com/python/cpython/issues/106363
    """

    _fields: ClassVar[tuple[str, ...]] = ()
    r"""The fields of the namedtuple."""

    def __instancecheck__(cls, instance: object) -> TypeIs["NTuple"]:  # noqa: N805
        return cls.__subclasscheck__(type(instance))

    def __subclasscheck__(cls, subclass: type) -> TypeIs[type["NTuple"]]:  # noqa: N805
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

    # NOTE: Added Final to silence pyright v1.1.376 complaints.
    # FIXME: python=3.13 use Final[ClassVar[tuple[str, ...]]].
    _fields: Final[tuple[str, ...]]  # type: ignore[misc]
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


class _SlottedMeta(ProtocolMeta):
    r"""Metaclass for `Slotted`.

    FIXME: https://github.com/python/cpython/issues/112319
    This issue will make the need for metaclass obsolete.
    """

    def __instancecheck__(cls, instance: object) -> TypeIs["Slotted"]:  # noqa: N805
        return cls.__subclasscheck__(type(instance))

    def __subclasscheck__(cls, subclass: type) -> TypeIs[type["Slotted"]]:  # noqa: N805
        slots = getattr(subclass, "__slots__", None)
        return isinstance(slots, str | Iterable)


@runtime_checkable
class Slotted(Protocol, metaclass=_SlottedMeta):
    r"""Protocol for objects that are slotted."""

    __slots__: ClassVar[tuple[str, ...]] = ()


def issubclass_dataclass(cls: type, /) -> TypeIs[type[Dataclass]]:
    return issubclass(cls, Dataclass)  # type: ignore[misc]


def isinstance_dataclass(obj: object, /) -> TypeIs[Dataclass]:
    return issubclass(type(obj), Dataclass)  # type: ignore[misc]


@overload
def is_dataclass(obj: type, /) -> TypeIs[type[Dataclass]]: ...
@overload
def is_dataclass(obj: object, /) -> TypeIs[Dataclass]: ...
def is_dataclass(obj: object, /) -> TypeIs[Dataclass] | TypeIs[type[Dataclass]]:
    r"""Check if the object is a dataclass."""
    if isinstance(obj, type):
        return issubclass(obj, Dataclass)  # type: ignore[misc]
    return issubclass(type(obj), Dataclass)  # type: ignore[misc]


def issubclass_namedtuple(cls: type, /) -> TypeIs[type[NTuple]]:
    return issubclass(cls, NTuple)  # type: ignore[misc]


def isinstance_namedtuple(obj: object, /) -> TypeIs[NTuple]:
    return issubclass(type(obj), NTuple)  # type: ignore[misc]


@overload
def is_namedtuple(obj: type, /) -> TypeIs[type[NTuple]]: ...
@overload
def is_namedtuple(obj: object, /) -> TypeIs[NTuple]: ...
def is_namedtuple(obj: object, /) -> TypeIs[NTuple] | TypeIs[type[NTuple]]:
    r"""Check if the object is a namedtuple."""
    if isinstance(obj, type):
        return issubclass(obj, NTuple)  # type: ignore[misc]
    return issubclass(type(obj), NTuple)  # type: ignore[misc]


def is_slotted(obj: object, /) -> TypeIs[Slotted]:
    r"""Check if the object is slotted."""
    return hasattr(obj, "__slots__")


# endregion generic factory-protocols --------------------------------------------------
