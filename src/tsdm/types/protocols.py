r"""Protocols used in tsdm.

References
----------
- https://www.python.org/dev/peps/pep-0544/
- https://docs.python.org/3/library/typing.html#typing.Protocol
- https://numpy.org/doc/stable/reference/c-api/array.html
"""

__all__ = [
    # Classes
    "Hash",
    "Lookup",
    "ShapeLike",
    # Mixins
    "SupportsArray",
    "SupportsDataframe",
    "SupportsDevice",
    "SupportsDtype",
    "SupportsNdim",
    "SupportsShape",
    # Arrays
    "SeriesKind",
    "TableKind",
    "ArrayKind",
    "NumericalArray",
    "MutableArray",
    # Time-Types
    # stdlib
    "SupportsGetItem",
    "SupportsKeysAndGetItem",
    "SupportsKwargs",
    "MappingProtocol",
    "MutableMappingProtocol",
    "SequenceProtocol",
    "MutableSequenceProtocol",
    # Factory classes
    "Dataclass",
    "NTuple",
    "is_dataclass",
    "is_namedtuple",
    # Functions
    "assert_protocol",
    # TypeVars
    "ArrayType",
    "NumericalArrayType",
    "MutableArrayType",
]

import dataclasses
from abc import abstractmethod
from collections.abc import (
    Collection,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Reversible,
    ValuesView,
)
from typing import (
    Any,
    ClassVar,
    NamedTuple,
    ParamSpec,
    Protocol,
    Self,
    SupportsIndex,
    TypeGuard,
    TypeVar,
    overload,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray
from typing_extensions import get_original_bases, get_protocol_members

from tsdm.types.variables import (
    any_co as T_co,
    any_var as T,
    key_contra,
    key_var as K,
    scalar_co,
    value_co as V_co,
    value_var as V,
)

ArrayType = TypeVar("ArrayType", bound="ArrayKind")
NumericalArrayType = TypeVar("NumericalArrayType", bound="NumericalArray")
MutableArrayType = TypeVar("MutableArrayType", bound="MutableArray")
P = ParamSpec("P")
Scalar = TypeVar("Scalar")
BoolArray = TypeVar("BoolArray", bound="NumericalArray", covariant=True)


def assert_protocol(obj: Any, proto: type, /) -> None:
    """Assert that the object is a given protocol."""
    if isinstance(obj, type):
        match = issubclass(obj, proto)
        name = obj.__name__
    else:
        match = isinstance(obj, proto)
        name = obj.__class__.__name__

    if not match:
        raise AssertionError(
            f"{name} is not a {proto.__name__}!"
            f"\n Missing Attributes: {get_protocol_members(proto) - set(dir(obj))}"
        )


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
class ShapeLike(Protocol):
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
    def __add__(self, other: Self | tuple, /) -> "ShapeLike": ...
    def __eq__(self, other: object, /) -> bool: ...
    def __ne__(self, other: object, /) -> bool: ...
    def __lt__(self, other: Self | tuple, /) -> bool: ...
    def __le__(self, other: Self | tuple, /) -> bool: ...


@runtime_checkable
class SupportsShape(Protocol[scalar_co]):
    r"""We just test for shape, since e.g. tf.Tensor does not have ndim."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Yield the shape of the array."""
        ...


@runtime_checkable
class SupportsDtype(Protocol):
    r"""We just test for dtype, since e.g. tf.Tensor does not have ndim."""

    @property
    def dtype(self) -> str | np.dtype | type:
        r"""Yield the data type of the array."""
        ...


@runtime_checkable
class SupportsNdim(Protocol):
    r"""We just test for ndim, since e.g. tf.Tensor does not have ndim."""

    @property
    def ndim(self) -> int:
        r"""Number of dimensions."""
        ...


@runtime_checkable
class SupportsDevice(Protocol):
    """Protocol for objects that support `device`."""

    @property
    def device(self) -> Any:
        """Return the device of the tensor."""
        ...


@runtime_checkable
class SupportsArray(Protocol):
    """Protocol for objects that support `__array__`.

    See: https://numpy.org/doc/stable/reference/c-api/array.html
    """

    def __array__(self) -> NDArray[np.object_]:
        """Return the array of the tensor."""
        ...


@runtime_checkable
class SupportsDataframe(Protocol):
    """Protocol for objects that support `__dataframe__`.

    See: https://data-apis.org/dataframe-protocol/latest/index.html
    """

    def __dataframe__(self) -> object:
        """Return the dataframe of the tensor."""
        ...


@runtime_checkable
class SeriesKind(Protocol[scalar_co]):
    """A 1d array of homogeneous data type.

    Examples:
        - `pandas.Series`
        - `polars.Series`
        - `pyarrow.Array`

    Counter-Examples:
        - `numpy.ndarray`  lacks `value_counts`
        - `pandas.DataFrame`  lacks `value_counts`
        - `polars.DataFrame`  lacks `value_counts`
        - `pyarrow.Table`  lacks `value_counts`
        - `torch.Tensor`  lacks `value_counts`
    """

    def __array__(self) -> NDArray:
        """Return a numpy array (usually of type `object` since columns are heterogeneous).

        References
            - https://numpy.org/devdocs/user/basics.interoperability.html
        """
        ...

    def __len__(self) -> int:
        """Number of elements in the series."""
        ...

    def __iter__(self) -> Iterator[scalar_co]:
        """Iterate over the elements of the series."""
        ...

    @overload
    def __getitem__(self, key: int, /) -> scalar_co: ...
    @overload
    def __getitem__(self, key: slice, /) -> Self: ...

    def unique(self) -> Self:
        """Return the unique elements of the series."""
        ...

    # NOTE: kind of inconsistent across backends
    def value_counts(self) -> Self | "SeriesKind" | "TableKind":
        """Return the number of occurences for each unique element of the series."""
        ...

    # NOTE: kind of inconsistent across backends
    def take(self, indices: Self | list[int] | NDArray, /) -> Self:
        """Select elements from the series by index."""
        ...


@runtime_checkable
class TableKind(Protocol):
    """A 2d column-oriented array with heterogenous data types.

    That it, it is a column-oriented 2d tensor which allows heterogenous data types.

    Note:
        In contrast to tensors (row-oriented), tables are column-oriented. Therefore,
        `__getitem__` returns a column, which is a SeriesKind, i.e. homogeneous 1d tensor.

    Examples:
        - `pandas.DataFrame`
        - `polars.DataFrame`
        - `pyarrow.Table`

    Counter-Examples:
        - `numpy.ndarray`  lacks `__dataframe__`
        - `pandas.Series`  lacks `__dataframe__`
        - `polars.Series`  lacks `__dataframe__`
        - `pyarrow.Array`  lacks `__dataframe__`
        - `torch.Tensor`  lacks `__dataframe__`
    """

    @property
    def shape(self) -> tuple[int, int]:
        """Yield the shape of the table."""
        ...

    def __array__(self) -> NDArray[np.object_]:
        """Return a numpy array (usually of type `object` since columns are heterogeneous).

        References
            - https://numpy.org/devdocs/user/basics.interoperability.html
        """
        ...

    def __dataframe__(
        self, nan_as_null: bool = False, allow_copy: bool = True
    ) -> object:
        """Return the dataframe interchange object implementing the interchange protocol.

        References:
            - https://data-apis.org/dataframe-protocol/latest/index.html
        """
        ...

    def __len__(self) -> int:
        """Number of rows."""
        ...

    def __getitem__(self, key: Any, /) -> "SeriesKind" | Self:
        """Return the corresponding column."""
        ...

    # FIXME: https://github.com/pola-rs/polars/issues/11911
    # def take(self, indices: Any, /) -> Any:
    #     """Return an element/slice of the table."""
    #     ...

    # @property
    # def columns(self) -> tuple[str, ...]:
    #     """Yield the columns or the column names."""
    #     ...
    #
    # def __iter__(self) -> Iterator[str]:
    #     """Iterate over the column or column names."""
    #     ...


@runtime_checkable
class ArrayKind(Protocol[scalar_co]):
    r"""An n-dimensional array of a single homogeneous data type.

    Examples:
        - `numpy.ndarray`
        - `pandas.Series`
        - `polars.Series`
        - `pyarrow.Array`
        - `tensorflow.Tensor`
        - `torch.Tensor`
        - `memoryview`

    Counter-Examples:
        - `pandas.DataFrame`
        - `polars.DataFrame`
        - `pyarrow.Table`

    References:
        - https://docs.python.org/3/c-api/buffer.html
        - https://numpy.org/doc/stable/reference/arrays.interface.html
    """

    # NOTE: This is a highly cut down version, to support the bare minimum.

    @property
    def shape(self) -> tuple[int, ...]:
        """Yield the shape of the array."""
        ...

    # @property
    # def ndim(self) -> int:
    #     """Number of dimensions."""
    #     ...

    def __array__(self) -> NDArray:
        """Return a numpy array.

        References
            - https://numpy.org/devdocs/user/basics.interoperability.html
        """
        ...

    def __len__(self) -> int:
        """Number of elements along the first axis."""
        ...

    def __getitem__(self, key: Any, /) -> Self | scalar_co:
        """Return an element/slice of the table."""
        ...

    def take(self, indices: Any, /) -> Any:
        """Return an element/slice of the table."""
        ...

    # @property
    # def ndim(self) -> int:
    #     r"""Number of dimensions."""
    #     ...

    # @property
    # def shape(self) -> tuple[int, ...]:
    #     """Yield the shape of the array."""
    #     ...

    # def __iter__(self) -> Iterator[Self]:
    #     """Iterate over the first dimension."""
    #     ...

    # @overload
    # def __getitem__(self, key, /):
    # def __getitem__(self, key: None | slice | list | tuple | Self, /) -> Self:
    #     """Return an element/slice of the array."""
    #     ...


@runtime_checkable
class NumericalArray(ArrayKind[Scalar], Protocol[Scalar]):
    r"""Subclass of `Array` that supports numerical operations.

    Examples:
        - `numpy.ndarray`
        - `pandas.Series`
        - `pandas.DataFrame`
        - `torch.Tensor`
        - `polars.Series`

    Counter-Examples:
        - `polars.DataFrame`
        - `pyarrow.Array`
        - `pyarrow.Table`
    """

    def all(self) -> Self | bool:
        """Return True if all elements are True."""
        ...

    def any(self) -> Self | bool:
        """Return True if any element is True."""
        ...

    # unary operations
    # positive +
    def __pos__(self) -> Self: ...

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

    # less than or equal <=
    def __le__(self, other: Self | Scalar, /) -> Self: ...

    # greater than or equal >=
    def __ge__(self, other: Self | Scalar, /) -> Self: ...

    # less than <
    def __lt__(self, other: Self | Scalar, /) -> Self: ...

    # greater than >
    def __gt__(self, other: Self | Scalar, /) -> Self: ...

    # matrix multiplication @
    # def __matmul__(self, other: Self, /) -> Self: ...
    # def __rmatmul__(self, other: Self, /) -> Self: ...

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
    def __pow__(self, exponent: Self | Scalar, /) -> Self: ...
    def __rpow__(self, base: Self | Scalar, /) -> Self: ...

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


@runtime_checkable
class MutableArray(NumericalArray[Scalar], Protocol[Scalar]):
    r"""Subclass of `Array` that supports inplace operations.

    Examples:
        - `numpy.ndarray`
        - `pandas.Series`
        - `pandas.DataFrame`
        - `torch.Tensor`

    Counter-Examples:
        - `polars.DataFrame`
        - `polars.Series`
        - `pyarrow.Array`
        - `pyarrow.Table`
    """

    # inplace matrix multiplication @=
    # NOTE: __imatmul__ not supported since it potentially changes the shape of the array.
    # def __imatmul__(self, other: Self, /) -> Self: ...

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
    def __ipow__(self, power: Self | Scalar, /) -> Self: ...

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


@runtime_checkable
class SupportsGetItem(Protocol[key_contra, V_co]):
    """Protocol for objects that support `__getitem__`."""

    def __getitem__(self, key: key_contra, /) -> V_co: ...


@runtime_checkable
class SupportsKeysAndGetItem(Protocol[K, V_co]):
    """Protocol for objects that support `__getitem__` and `keys`."""

    def keys(self) -> Iterable[K]: ...
    def __getitem__(self, key: K, /) -> V_co: ...


class SupportsKwargsType(type(Protocol)):  # type: ignore[misc]
    """Metaclass for `SupportsKwargs`."""

    def __instancecheck__(self, other: object, /) -> bool:
        return isinstance(other, SupportsKeysAndGetItem) and isinstance(
            next(iter(other.keys())), str
        )

    def __subclasscheck__(self, other: type, /) -> bool:
        raise NotImplementedError("Cannot check whether a class is a SupportsKwargs.")


@runtime_checkable
class SupportsKwargs(Protocol[V_co], metaclass=SupportsKwargsType):
    """Protocol for objects that support `**kwargs`."""

    def keys(self) -> Iterable[str]: ...
    def __getitem__(self, key: str, /) -> V_co: ...


@runtime_checkable
class SequenceProtocol(Collection[T_co], Reversible[T_co], Protocol[T_co]):
    """Protocol version of `collections.abc.Sequence`."""

    # FIXME: implement mixin methods

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


@runtime_checkable
class MutableSequenceProtocol(SequenceProtocol[T], Protocol[T]):
    """Protocol version of `collections.abc.MutableSequence`."""

    # FIXME: implement mixin methods

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


@runtime_checkable
class MappingProtocol(Collection[K], Protocol[K, V_co]):
    """Protocol version of `collections.abc.Mapping`."""

    # FIXME: implement mixin methods

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


@runtime_checkable
class MutableMappingProtocol(MappingProtocol[K, V], Protocol[K, V]):
    """Protocol version of `collections.abc.MutableMapping`."""

    # FIXME: implement mixin methods

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


# region generic factory-protocols -----------------------------------------------------


@runtime_checkable
class Dataclass(Protocol):
    r"""Protocol for anonymous dataclasses.

    Similar to `DataClassInstance` from typeshed, but allows isinstance and issubclass.
    """

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]
    r"""The fields of the dataclass."""

    # @property
    # def __dataclass_fields__(self) -> dict[str, dataclasses.Field]:
    #     """Return the fields of the dataclass."""
    #     ...

    @classmethod
    def __subclasshook__(cls, other: type, /) -> bool:
        """Cf https://github.com/python/cpython/issues/106363."""
        fields = getattr(other, "__dataclass_fields__", None)
        return isinstance(fields, dict)


@runtime_checkable
class NTuple(Protocol[T_co]):  # FIXME: Use TypeVarTuple
    r"""Protocol for anonymous namedtuple.

    References:
        - https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi
    """

    # NOTE: Problems is set tuple[str, ...]
    _fields: Any  # FIXME: Should be tuple[*str] (not tuple[str, ...])
    r"""The fields of the namedtuple."""
    # FIXME: https://github.com/python/typing/issues/1216
    # FIXME: https://github.com/python/typing/issues/1273

    # @property  # NOTE: @property won't work for classes
    # def _fields(self) -> tuple[str, ...]:
    #     """Return the fields of the namedtuple."""
    #     ...

    # fmt: off
    def _asdict(self) -> dict[str, Any]: ...
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

    @classmethod
    def __subclasshook__(cls, other: type, /) -> bool:
        """Cf https://github.com/python/cpython/issues/106363."""
        if NamedTuple in get_original_bases(other):
            return True
        return NotImplemented


@overload
def is_dataclass(obj: type, /) -> TypeGuard[type[Dataclass]]: ...
@overload
def is_dataclass(obj: object, /) -> TypeGuard[Dataclass]: ...
def is_dataclass(obj: Any, /) -> TypeGuard[Dataclass | type[Dataclass]]:
    r"""Check if the object is a dataclass."""
    # return dataclasses.is_dataclass(obj)
    return (
        issubclass(obj, Dataclass)  # type: ignore[misc]
        if isinstance(obj, type)
        else issubclass(type(obj), Dataclass)  # type: ignore[misc]
    )


# @overload
# def is_dataclass(obj: DataclassInstance) -> Literal[True]: ...
# @overload
# def is_dataclass(obj: type) -> TypeGuard[type[DataclassInstance]]: ...
# @overload
# def is_dataclass(
#     obj: object,
# ) -> TypeGuard[DataclassInstance | type[DataclassInstance]]: ...


#
#
# reveal_type(is_dataclass)


@overload
def is_namedtuple(obj: type, /) -> TypeGuard[type[NTuple]]: ...
@overload
def is_namedtuple(obj: object, /) -> TypeGuard[NTuple]: ...
def is_namedtuple(obj: Any, /) -> TypeGuard[NTuple | type[NTuple]]:
    """Check if the object is a namedtuple."""
    return (
        issubclass(obj, NTuple)  # type: ignore[misc]
        if isinstance(obj, type)
        else issubclass(type(obj), NTuple)  # type: ignore[misc]
    )


# NOTE: TypedDict not supported, since instances of type-dicts forget their parent class.
#  and are just plain dicts.

# endregion generic factory-protocols --------------------------------------------------
