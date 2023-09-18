r"""Protocols used in tsdm.

References
----------
- https://www.python.org/dev/peps/pep-0544/
- https://docs.python.org/3/library/typing.html#typing.Protocol
- https://numpy.org/doc/stable/reference/c-api/array.html
"""

__all__ = [
    # Classes
    "Array",
    "NumericalArray",
    "SupportsShape",
    "Dataclass",
    "Hash",
    "Lookup",
    "NTuple",
    "Shape",
    # Functions
    "is_dataclass",
]

import dataclasses
from collections.abc import Iterator, Mapping, Sequence
from types import EllipsisType
from typing import (
    Any,
    NamedTuple,
    Protocol,
    TypeGuard,
    TypeVar,
    get_type_hints,
    overload,
    runtime_checkable,
)

import numpy
from numpy.typing import NDArray
from typing_extensions import Self, SupportsIndex, get_original_bases

from tsdm.types.variables import (
    any_co as T_co,
    int_var,
    key_contra,
    scalar_co,
    value_co,
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
    def __eq__(self, other: Self | tuple, /) -> bool: ...
    def __ne__(self, other: Self | tuple, /) -> bool: ...
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

    def __array__(self) -> NDArray[scalar_co]:
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
    def dtype(self) -> scalar_co | type[scalar_co] | numpy.dtype[scalar_co]:
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
    @overload
    def __getitem__(self, key: None | slice | list | tuple | Self, /) -> Self: ...
    def __getitem__(self, key, /):
        """Return an element/slice of the array."""
        ...


print(get_type_hints(Array))

# # fmt: off
# def __len__(self) -> int: ...
# # @overload
# # def __getitem__(self: A, key: int) -> Any: ...
# # @overload
# # def __getitem__(self: A, key: Sequence[bool]) -> A: ...
# # @overload
# # def __getitem__(self: A, key: Sequence[int]) -> A: ...
# def __getitem__(self: A, key: Any) -> A: ...
# def __eq__(self: A, other: Any) -> A: ...  # type: ignore[override]
# def __le__(self: A, other: Any) -> A: ...
# def __ge__(self: A, other: Any) -> A: ...
# def __lt__(self: A, other: Any) -> A: ...
# def __gt__(self: A, other: Any) -> A: ...
# def __ne__(self: A, other: Any) -> A: ...  # type: ignore[override]
# def __neg__(self: A) -> A: ...
# def __invert__(self: A) -> A: ...
# def __add__(self: A, other: Any) -> A: ...
# def __radd__(self: A, other: Any) -> A: ...
# def __iadd__(self: A, other: Any) -> A: ...
# def __sub__(self: A, other: Any) -> A: ...
# def __rsub__(self: A, other: Any) -> A: ...
# def __isub__(self: A, other: Any) -> A: ...
# def __mul__(self: A, other: Any) -> A: ...
# def __rmul__(self: A, other: Any) -> A: ...
# def __imul__(self: A, other: Any) -> A: ...
# def __truediv__(self: A, other: Any) -> A: ...
# def __rtruediv__(self: A, other: Any) -> A: ...
# def __itruediv__(self: A, other: Any) -> A: ...
# # fmt: on

# [
#     "__abs__",
#     "__add__",
#     "__and__",
#     "__bool__",
#     "__dir__",
#     "__eq__",
#     "__float__",
#     "__floordiv__",
#     "__ge__",
#     "__gt__",
#     "__hash__",
#     "__int__",
#     "__invert__",
#     "__le__",
#     "__lt__",
#     "__matmul__",
#     "__mod__",
#     "__mul__",
#     "__ne__",
#     "__neg__",
#     "__or__",
#     "__pow__",
#     "__radd__",
#     "__rand__",
#     "__reduce__",
#     "__reduce_ex__",
#     "__repr__",
#     "__rfloordiv__",
#     "__rmatmul__",
#     "__rmod__",
#     "__rmul__",
#     "__ror__",
#     "__rpow__",
#     "__rsub__",
#     "__rtruediv__",
#     "__rxor__",
#     "__sizeof__",
#     "__str__",
#     "__sub__",
#     "__truediv__",
#     "__xor__",
# ]


class NumericalArray(Array[scalar_co]):
    r"""Protocol for array-like objects.

    Compared to a Table (e.g. `pandas.DataFrame` / `pyarrow.Table`), an Array has a single dtype.
    """

    @property
    def ndim(self) -> int:
        r"""Number of dimensions."""
        ...

    @property
    def dtype(self) -> scalar_co:
        r"""Yield the data type of the array."""
        ...

    @property
    def shape(self) -> Sequence[int]:
        """Yield the shape of the array."""
        ...

    def __len__(self) -> int:
        """Number of elements along first axis."""
        ...

    def __getitem__(self, key: Any) -> Self:
        """Return an element/slice of the table."""
        ...

    # Numerical Operations
    # fmt: off
    def __neg__(self) -> Self: ...
    def __invert__(self) -> Self: ...
    # comparisons
    def __eq__(self, other: Self, /) -> Self: ...  # type: ignore[override]
    def __le__(self, other: Self, /) -> Self: ...
    def __ge__(self, other: Self, /) -> Self: ...
    def __lt__(self, other: Self, /) -> Self: ...
    def __gt__(self, other: Self, /) -> Self: ...
    def __ne__(self, other: Self, /) -> Self: ...  # type: ignore[override]
    # arithmetic
    def __add__(self, other: Self, /) -> Self: ...
    def __radd__(self, other: Self, /) -> Self: ...
    def __iadd__(self, other: Self, /) -> Self: ...
    def __sub__(self, other: Self, /) -> Self: ...
    def __rsub__(self, other: Self, /) -> Self: ...
    def __isub__(self, other: Self, /) -> Self: ...
    def __mul__(self, other: Self, /) -> Self: ...
    def __rmul__(self, other: Self, /) -> Self: ...
    def __imul__(self, other: Self, /) -> Self: ...
    def __truediv__(self, other: Self, /) -> Self: ...
    def __rtruediv__(self, other: Self, /) -> Self: ...
    def __itruediv__(self, other: Self, /) -> Self: ...
    def __pow__(self, power: Self, modulo: None | int = None, /) -> Self: ...
    # boolean
    def __lshift__(self, other: Self, /) -> Self: ...
    def __rlshift__(self, other: Self, /) -> Self: ...
    def __ilshift__(self, other: Self, /) -> Self: ...
    def __rshift__(self, other: Self, /) -> Self: ...
    def __rrshift__(self, other: Self, /) -> Self: ...
    def __irshift__(self, other: Self, /) -> Self: ...
    def __and__(self, other: Self, /) -> Self: ...
    def __rand__(self, other: Self, /) -> Self: ...
    def __iand__(self, other: Self, /) -> Self: ...
    def __or__(self, other: Self, /) -> Self: ...
    def __ror__(self, other: Self, /) -> Self: ...
    def __ior__(self, other: Self, /) -> Self: ...
    def __xor__(self, other: Self, /) -> Self: ...
    def __rxor__(self, other: Self, /) -> Self: ...
    def __ixor__(self, other: Self, /) -> Self: ...
    # fmt: on


@runtime_checkable
class Lookup(Protocol[key_contra, value_co]):
    """Mapping/Sequence like generic that is contravariant in Keys."""

    def __contains__(self, key: key_contra) -> bool:
        # Here, any Hashable input is accepted.
        """Return True if the map contains the given key."""
        ...

    def __getitem__(self, key: key_contra) -> value_co:
        """Return the value associated with the given key."""
        ...

    def __len__(self) -> int:
        """Return the number of items in the map."""
        ...


# endregion container protocols --------------------------------------------------------


# region misc protocols ----------------------------------------------------------------
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
