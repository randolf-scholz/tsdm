r"""Protocols used in tsdm.

References
----------
- https://www.python.org/dev/peps/pep-0544/
- https://docs.python.org/3/library/typing.html#typing.Protocol
- https://numpy.org/doc/stable/reference/c-api/array.html
"""

__all__ = [
    # Classes
    "SupportsShape",
    "Dataclass",
    "Hash",
    "Lookup",
    "NTuple",
    "Shape",
    # Arrays
    "Array",
    "NumericalArray",
    "MutableArray",
    # stdlib
    "SupportsKeysAndGetItem",
    "MappingProtocol",
    "MutableMappingProtocol",
    "SequenceProtocol",
    "MutableSequenceProtocol",
    # Functions
    "is_dataclass",
]

import dataclasses
from abc import abstractmethod
from collections.abc import (
    Collection,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    Reversible,
    Sequence,
    ValuesView,
)
from typing import (
    Any,
    NamedTuple,
    Protocol,
    TypeGuard,
    TypeVar,
    overload,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self, SupportsIndex, get_original_bases

from tsdm.types.variables import (
    any_co as T_co,
    any_var as T,
    key_contra,
    key_var as K,
    scalar_co,
    scalar_var as Scalar,
    value_co as V_co,
    value_var as V,
)

A = TypeVar("A", bound="Array")


# region generic factory-protocols -----------------------------------------------------
@runtime_checkable
class Dataclass(Protocol):
    r"""Protocol for anonymous dataclasses."""

    @property
    def __dataclass_fields__(self) -> Mapping[str, dataclasses.Field]:
        r"""Return the fields of the dataclass."""
        ...


def is_dataclass(obj: Any) -> TypeGuard[Dataclass]:
    r"""Check if object is a dataclass."""
    return dataclasses.is_dataclass(obj)


@runtime_checkable
class NTuple(Protocol[T_co]):
    r"""Protocol for anonymous namedtuple.

    References:
        - https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi
    """

    @classmethod
    def __subclasshook__(cls, other: type, /) -> bool:
        """Cf https://github.com/python/cpython/issues/106363."""
        if NamedTuple in get_original_bases(other):
            return True
        return NotImplemented

    # fmt: off
    @property
    def _fields(self) -> tuple[str, ...]: ...
    def _asdict(self) -> Mapping[str, T_co]: ...
    # def __new__(cls, __iterable: Iterable[T_co] = ...) -> Self: ...
    def __len__(self) -> int: ...
    def __contains__(self, key: object, /) -> bool: ...
    @overload
    def __getitem__(self, key: SupportsIndex, /) -> T_co: ...
    @overload
    def __getitem__(self, key: slice, /) -> tuple[T_co, ...]: ...
    def __iter__(self) -> Iterator[T_co]: ...
    # def __lt__(self, __value: tuple[T_co, ...]) -> bool: ...
    # def __le__(self, __value: tuple[T_co, ...]) -> bool: ...
    # def __gt__(self, __value: tuple[T_co, ...]) -> bool: ...
    # def __ge__(self, __value: tuple[T_co, ...]) -> bool: ...
    # @overload
    # def __add__(self, __value: tuple[T_co, ...]) -> tuple[T_co, ...]: ...
    # @overload
    # def __add__(self, __value: tuple[T, ...]) -> tuple[T_co | T, ...]: ...
    # def __mul__(self, __value: SupportsIndex) -> tuple[T_co, ...]: ...
    # def __rmul__(self, __value: SupportsIndex) -> tuple[T_co, ...]: ...
    # def count(self, __value: Any) -> int: ...
    # def index(self, __value: Any, __start: SupportsIndex = 0, __stop: SupportsIndex = sys.maxsize) -> int: ...
    # fmt: on


# endregion generic factory-protocols --------------------------------------------------


# region misc protocols ----------------------------------------------------------------
@runtime_checkable
class Lookup(Protocol[key_contra, V_co]):
    """Mapping/Sequence like generic that is contravariant in Keys."""

    def __contains__(self, key: key_contra) -> bool:
        # Here, any Hashable input is accepted.
        """Return True if the map contains the given key."""
        ...

    def __getitem__(self, key: key_contra) -> V_co:
        """Return the value associated with the given key."""
        ...

    def __len__(self) -> int:
        """Return the number of items in the map."""
        ...


@runtime_checkable
class Hash(Protocol):
    """Protocol for hash-functions."""

    name: str

    def digest_size(self) -> int:
        """Return the size of the hash in bytes."""
        ...

    def block_size(self) -> int:
        """Return the internal block size of the hash in bytes."""
        ...

    def update(self, data: bytes) -> None:
        """Update this hash object's state with the provided string."""
        ...

    def digest(self) -> bytes:
        """Return the digest value as a string of binary data."""
        ...

    def hexdigest(self) -> str:
        """Return the digest value as a string of hexadecimal digits."""
        ...

    def copy(self) -> Self:
        """Return a clone of the hash object."""
        ...


# endregion misc protocols -------------------------------------------------------------


# region container protocols -----------------------------------------------------------
@runtime_checkable
class Shape(Protocol):
    """Protocol for shapes, very similar to tuple, but without `__contains__`.

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
    # NOTE: Not returning self, cf. https://github.com/python/typeshed/issues/10727
    def __add__(self, other: Self | tuple, /) -> "Shape": ...
    def __eq__(self, other: object, /) -> bool: ...
    def __ne__(self, other: object, /) -> bool: ...
    def __lt__(self, other: Self | tuple, /) -> bool: ...
    def __le__(self, other: Self | tuple, /) -> bool: ...


@runtime_checkable
class SupportsShape(Protocol[scalar_co]):
    r"""We just test for shape, since e.g. tf.Tensor does not have ndim."""

    @property
    def shape(self) -> Sequence[int]:
        """Yield the shape of the array."""
        ...

    def __len__(self) -> int:
        """Number of elements along first axis."""
        ...

    # binary operations
    def __getitem__(self, key: Any, /) -> Self | scalar_co:
        """Return an element/slice of the table."""
        ...


@runtime_checkable
class Array(Protocol[scalar_co]):
    r"""Protocol for array-like objects (tensors with single data type).

    Matches with

    - `numpy.ndarray`
    - `torch.Tensor`
    - `tensorflow.Tensor`
    - `jax.numpy.ndarray`
    - `pandas.Series`

    Does not match with

    - `pandas.DataFrame`
    - `pyarrow.Table`
    - `pyarrow.Array`
    """

    def __array__(self) -> NDArray:
        """Return a numpy array.

        References
            - https://numpy.org/devdocs/user/basics.interoperability.html
        """
        ...

    @property
    def ndim(self) -> int:
        r"""Number of dimensions."""
        ...

    @property
    def dtype(self) -> scalar_co | np.dtype | type:
        r"""Yield the data type of the array."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Yield the shape of the array."""
        ...

    def __len__(self) -> int:
        """Number of elements along first axis."""
        ...

    def __iter__(self) -> Iterator[Self]:
        """Iterate over the first dimension."""
        ...

    # binary operations
    # @overload
    # def __getitem__(self, key, /):
    def __getitem__(self, key: None | slice | list | tuple | Self, /) -> Self:
        """Return an element/slice of the array."""
        ...


class NumericalArray(Array[Scalar], Protocol[Scalar]):
    r"""Subclass of `Array` that supports numerical operations.

    Note:
        - incplace operations are excluded since certain libraries make tensors immutable.
    """

    # unary operations
    # negation -
    def __neg__(self) -> Self: ...

    # bitwise NOT ~
    def __invert__(self) -> Self: ...

    # absolute value abs()
    def __abs__(self) -> Self: ...

    # comparisons
    # equality ==
    def __eq__(self, other: Self | Scalar, /) -> Self: ...  # type: ignore[override]

    # inequality !=
    def __ne__(self, other: Self | Scalar, /) -> Self: ...  # type: ignore[override]

    # greater than or equal >=
    def __le__(self, other: Self | Scalar, /) -> Self: ...

    # less than or equal <=
    def __ge__(self, other: Self | Scalar, /) -> Self: ...

    # less than <
    def __lt__(self, other: Self | Scalar, /) -> Self: ...

    # greater than >
    def __gt__(self, other: Self | Scalar, /) -> Self: ...

    # matrix operations
    # matmul @
    def __matmul__(self, other: Self, /) -> Self: ...
    def __rmatmul__(self, other: Self, /) -> Self: ...

    # arithmetic
    # addition +
    def __add__(self, other: Self | Scalar, /) -> Self: ...
    def __radd__(self, other: Self | Scalar, /) -> Self: ...

    # subtraction -
    def __sub__(self, other: Self | Scalar, /) -> Self: ...
    def __rsub__(self, other: Self | Scalar, /) -> Self: ...

    # multiplication *
    def __mul__(self, other: Self | Scalar, /) -> Self: ...
    def __rmul__(self, other: Self | Scalar, /) -> Self: ...

    # true division /
    def __truediv__(self, other: Self | Scalar, /) -> Self: ...
    def __rtruediv__(self, other: Self | Scalar, /) -> Self: ...

    # floor division //
    def __floordiv__(self, other: Self | Scalar, /) -> Self: ...
    def __rfloordiv__(self, other: Self | Scalar, /) -> Self: ...

    # power **
    def __pow__(
        self, exponent: Self | Scalar, modulo: None | int = None, /
    ) -> Self: ...
    def __rpow__(self, base: Self | Scalar, modulo: None | int = None, /) -> Self: ...

    # modulo %
    def __mod__(self, other: Self | Scalar, /) -> Self: ...
    def __rmod__(self, other: Self | Scalar, /) -> Self: ...

    # boolean operations
    # AND &
    def __and__(self, other: Self | Scalar, /) -> Self: ...
    def __rand__(self, other: Self | Scalar, /) -> Self: ...

    # OR |
    def __or__(self, other: Self | Scalar, /) -> Self: ...
    def __ror__(self, other: Self | Scalar, /) -> Self: ...

    # XOR ^
    def __xor__(self, other: Self | Scalar, /) -> Self: ...
    def __rxor__(self, other: Self | Scalar, /) -> Self: ...

    # bitwise operators
    # # left shift <<
    # def __lshift__(self, other: Self | Scalar, /) -> Self: ...
    # def __rlshift__(self, other: Self | Scalar, /) -> Self: ...
    #
    # # right shift >>
    # def __rshift__(self, other: Self | Scalar, /) -> Self: ...
    # def __rrshift__(self, other: Self | Scalar, /) -> Self: ...


class MutableArray(NumericalArray[Scalar], Protocol[Scalar]):
    r"""Subclass of `Array` that supports inplace operations."""

    # inplace matrix operations
    # NOTE: __imatmul__ not supported since it potentially changes the shape of the array.
    # matmul @=
    # def __imatmul__(self, other: Self | Scalar, /) -> Self: ...

    # inplace arithmetic operations
    # addition +=
    def __iadd__(self, other: Self | Scalar, /) -> Self: ...

    # floor division //=
    def __ifloordiv__(self, other: Self | Scalar, /) -> Self: ...

    # modulo %=
    def __imod__(self, other: Self | Scalar, /) -> Self: ...

    # multiplication *=
    def __imul__(self, other: Self | Scalar, /) -> Self: ...

    # power **=
    def __ipow__(self, power: Self | Scalar, modulo: None | int = None, /) -> Self: ...

    # subtraction -=
    def __isub__(self, other: Self | Scalar, /) -> Self: ...

    # true division /=
    def __itruediv__(self, other: Self | Scalar, /) -> Self: ...

    # inplace boolean operations
    # AND &=
    def __iand__(self, other: Self | Scalar, /) -> Self: ...

    # OR |=
    def __ior__(self, other: Self | Scalar, /) -> Self: ...

    # XOR ^=
    def __ixor__(self, other: Self | Scalar, /) -> Self: ...

    # inplace bitwise operations
    # # left shift =<<
    # def __ilshift__(self, other: Self | Scalar, /) -> Self: ...
    #
    # # right shift =>>
    # def __irshift__(self, other: Self | Scalar, /) -> Self: ...


# endregion container protocols --------------------------------------------------------


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


class SupportsKeysAndGetItem(Protocol[K, V_co]):
    """Protocol for objects that support `__getitem__` and `keys`."""

    def keys(self) -> Iterable[K]: ...
    def __getitem__(self, __key: K) -> V_co: ...


class SequenceProtocol(Collection[T_co], Reversible[T_co], Protocol[T_co]):
    """Protocol version of `collections.abc.Sequence`."""

    @overload
    @abstractmethod
    def __getitem__(self, index: int) -> T_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, index: slice) -> "SequenceProtocol[T_co]": ...

    # Mixin methods
    def index(self, value: Any, start: int = 0, stop: int = ...) -> int: ...
    def count(self, value: Any) -> int: ...
    def __contains__(self, value: object) -> bool: ...
    def __iter__(self) -> Iterator[T_co]: ...
    def __reversed__(self) -> Iterator[T_co]: ...


class MutableSequenceProtocol(SequenceProtocol[T], Protocol[T]):
    """Protocol version of `collections.abc.MutableSequence`."""

    @abstractmethod
    def insert(self, index: int, value: T) -> None: ...
    @overload
    @abstractmethod
    def __getitem__(self, index: int) -> T: ...
    @overload
    @abstractmethod
    def __getitem__(self, index: slice) -> "MutableSequenceProtocol[T]": ...
    @overload
    @abstractmethod
    def __setitem__(self, index: int, value: T) -> None: ...
    @overload
    @abstractmethod
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...
    @overload
    @abstractmethod
    def __delitem__(self, index: int) -> None: ...
    @overload
    @abstractmethod
    def __delitem__(self, index: slice) -> None: ...

    # Mixin methods
    def append(self, value: T) -> None: ...
    def clear(self) -> None: ...
    def extend(self, values: Iterable[T]) -> None: ...
    def reverse(self) -> None: ...
    def pop(self, index: int = -1) -> T: ...
    def remove(self, value: T) -> None: ...
    def __iadd__(self, values: Iterable[T]) -> Self: ...


class MappingProtocol(Collection[K], Protocol[K, V_co]):
    """Protocol version of `collections.abc.Mapping`."""

    @abstractmethod
    def __getitem__(self, __key: K) -> V_co: ...

    # Mixin methods
    @overload
    def get(self, __key: K) -> V_co | None: ...
    @overload
    def get(self, __key: K, default: V_co | T) -> V_co | T: ...
    def items(self) -> ItemsView[K, V_co]: ...
    def keys(self) -> KeysView[K]: ...
    def values(self) -> ValuesView[V_co]: ...
    def __contains__(self, __key: object) -> bool: ...
    def __eq__(self, __other: object) -> bool: ...


class MutableMappingProtocol(MappingProtocol[K, V], Protocol[K, V]):
    """Protocol version of `collections.abc.MutableMapping`."""

    @abstractmethod
    def __setitem__(self, __key: K, __value: V) -> None: ...
    @abstractmethod
    def __delitem__(self, __key: K) -> None: ...

    # Mixin methods
    def clear(self) -> None: ...
    @overload
    def pop(self, __key: K) -> V: ...
    @overload
    def pop(self, __key: K, default: V) -> V: ...
    @overload
    def pop(self, __key: K, default: T) -> V | T: ...
    def popitem(self) -> tuple[K, V]: ...
    @overload
    def setdefault(
        self: "MutableMappingProtocol[K, T | None]", __key: K, __default: None = None
    ) -> T | None: ...
    @overload
    def setdefault(self, __key: K, __default: V) -> V: ...
    @overload
    def update(self, __m: SupportsKeysAndGetItem[K, V], **kwargs: V) -> None: ...
    @overload
    def update(self, __m: Iterable[tuple[K, V]], **kwargs: V) -> None: ...
    @overload
    def update(self, **kwargs: V) -> None: ...


# endregion stdlib protocols -----------------------------------------------------------
