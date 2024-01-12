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
    "VectorLike",
    # Mixins
    "SupportsArray",
    "SupportsDataframe",
    "SupportsDevice",
    "SupportsDtype",
    "SupportsItem",
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
    "MappingProtocol",
    "MutableMappingProtocol",
    "SequenceProtocol",
    "SupportsGetItem",
    "SupportsKeysAndGetItem",
    "SupportsKwargs",
    "SupportsLenAndGetItem",
    "MutableSequenceProtocol",
    # Factory classes
    "Dataclass",
    "NTuple",
    "is_dataclass",
    "is_namedtuple",
    # Functions
    # TypeVars
    "ArrayType",
    "TableType",
    "NumericalArrayType",
    "MutableArrayType",
]

import dataclasses
import typing
from abc import abstractmethod
from collections.abc import (
    Collection,
    Hashable,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    ValuesView,
)
from types import GenericAlias

import numpy as np
import typing_extensions
from numpy.typing import NDArray
from typing_extensions import (
    Any,
    ClassVar,
    ParamSpec,
    Protocol,
    Self,
    SupportsIndex,
    TypeGuard,
    TypeVar,
    get_original_bases,
    overload,
    runtime_checkable,
)

from tsdm.types.variables import K, K_contra, T, T_co, V, V_co, scalar_co

P = ParamSpec("P")
Scalar = TypeVar("Scalar")


# region misc protocols ----------------------------------------------------------------
@runtime_checkable
class GenericIterable(Protocol[T_co]):
    """Does not work currently!"""

    # FIXME: https://github.com/python/cpython/issues/112319
    def __class_getitem__(cls, item: type) -> GenericAlias: ...
    def __iter__(self) -> Iterator[T_co]: ...


@runtime_checkable
class VectorLike(Protocol[T_co]):
    """Alternative to `Sequence` without `__reversed__`, `index` and `count`.

    We remove these 3 sinc they are not present on certain vector data structures,
    for example, `__reversed__` is not present on `pandas.Index`.

    Examples:
        - list
        - tuple
        - numpy.ndarray
        - pandas.Index
        - pandas.Series (with integer index)
    """

    @abstractmethod
    def __len__(self) -> int: ...
    @overload
    @abstractmethod
    def __getitem__(self, index: int, /) -> T_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, index: slice, /) -> "VectorLike[T_co]": ...

    # Mixin methods
    # TODO: implement mixin methods
    def __iter__(self) -> Iterator[T_co]:
        for i in range(len(self)):
            yield self[i]

    def __contains__(self, value: object, /) -> bool:
        # NOTE: We need __contains__ to disallow `str`.
        return any(x == value or x is value for x in self)

    # def __reversed__(self) -> Iterator[T_co]:
    #     for i in reversed(range(len(self))):
    #         yield self[i]


@runtime_checkable
class Lookup(Protocol[K_contra, V_co]):
    """Mapping/Sequence like generic that is contravariant in Keys."""

    @abstractmethod
    def __contains__(self, key: K_contra, /) -> bool:
        # Here, any Hashable input is accepted.
        """Return True if the map contains the given key."""
        ...

    @abstractmethod
    def __getitem__(self, key: K_contra, /) -> V_co:
        """Return the value associated with the given key."""
        ...

    @abstractmethod
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
class SupportsDevice(Protocol):
    """Protocol for objects that support `device`."""

    @property
    @abstractmethod
    def device(self) -> Any:
        """Return the device of the tensor."""
        ...


@runtime_checkable
class SupportsDtype(Protocol):
    r"""We just test for dtype, since e.g. tf.Tensor does not have ndim."""

    @property
    @abstractmethod
    def dtype(self) -> str | np.dtype | type:
        r"""Yield the data type of the array."""
        ...


@runtime_checkable
class SupportsNdim(Protocol):
    r"""We just test for ndim, since e.g. tf.Tensor does not have ndim."""

    @property
    @abstractmethod
    def ndim(self) -> int:
        r"""Number of dimensions."""
        ...


@runtime_checkable
class SupportsShape(Protocol):
    r"""Protocol for objects that support shape."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Yield the shape of the array."""
        ...


@runtime_checkable
class SupportsArray(Protocol):
    """Protocol for objects that support `__array__`.

    See: https://numpy.org/doc/stable/reference/c-api/array.html
    """

    @abstractmethod
    def __array__(self) -> NDArray[np.object_]:
        """Return the array of the tensor."""
        ...


@runtime_checkable
class SupportsDataframe(Protocol):
    """Protocol for objects that support `__dataframe__`.

    See: https://data-apis.org/dataframe-protocol/latest/index.html
    """

    @abstractmethod
    def __dataframe__(self) -> object:
        """Return the dataframe of the tensor."""
        ...


@runtime_checkable
class SupportsItem(Protocol[scalar_co]):
    """Protocol for objects that support `.item()`."""

    @abstractmethod
    def item(self) -> scalar_co:
        """Return the scalar value the tensor if it only has a single element.

        If the tensor has more than one element, raise an error.
        """
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

    @abstractmethod
    def __array__(self) -> NDArray:
        """Return a numpy array (usually of type `object` since columns are heterogeneous).

        References:
            - https://numpy.org/devdocs/user/basics.interoperability.html
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Number of elements in the series."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[scalar_co]:
        """Iterate over the elements of the series."""
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: int, /) -> scalar_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice, /) -> Self: ...

    @abstractmethod
    def unique(self) -> Self:
        """Return the unique elements of the series."""
        ...

    # NOTE: kind of inconsistent across backends
    @abstractmethod
    def value_counts(self) -> Self | "SeriesKind" | "TableKind":
        """Return the number of occurences for each unique element of the series."""
        ...

    # NOTE: kind of inconsistent across backends
    @abstractmethod
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

        References:
            - https://numpy.org/devdocs/user/basics.interoperability.html
        """
        ...

    def __dataframe__(self, *, allow_copy: bool = True) -> object:
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

    @property
    def columns(self) -> Collection[Hashable] | Collection[SeriesKind]:
        """Yield the columns or the column names."""
        ...

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
        - `pandas.DataFrame` (different __getitem__)
        - `polars.DataFrame`
        - `pyarrow.Table`

    References:
        - https://docs.python.org/3/c-api/buffer.html
        - https://numpy.org/doc/stable/reference/arrays.interface.html
    """

    # NOTE: This is a highly cut down version, to support the bare minimum.

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Yield the shape of the array."""
        ...

    # @property
    # @abstractmethod
    # def ndim(self) -> int:
    #     """Number of dimensions."""
    #     ...

    @abstractmethod
    def __array__(self) -> NDArray:
        """Return a numpy array.

        References:
            - https://numpy.org/devdocs/user/basics.interoperability.html
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Number of elements along the first axis."""
        ...

    # @abstractmethod
    # def __iter__(self) -> Iterator[Self]:
    #     """Iterate over the first dimension."""
    #     ...

    @abstractmethod
    def __getitem__(self, key: Any, /) -> Self | scalar_co:
        """Return an element/slice of the table."""
        ...

    # @abstractmethod
    # def take(self, indices: Any, /) -> Any:
    #     """Return an element/slice of the table."""
    #     ...


@runtime_checkable
class NumericalArray(ArrayKind[Scalar], Protocol[Scalar]):
    r"""Subclass of `Array` that supports numerical operations.

    Examples:
        - `numpy.ndarray`
        - `pandas.Index`
        - `pandas.Series`
        - `pandas.DataFrame`
        - `torch.Tensor`

    Counter-Examples:
        - `polars.Series`  (missing ndim)
        - `polars.DataFrame`
        - `pyarrow.Array`
        - `pyarrow.Table`
    """

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Yield the shape of the array."""
        ...

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    @abstractmethod
    def __array__(self) -> NDArray:
        """Return a numpy array.

        References:
            - https://numpy.org/devdocs/user/basics.interoperability.html
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Number of elements along the first axis."""
        return int(self.shape[0])

    @abstractmethod
    def __iter__(self) -> Iterator[Self]:
        """Iterate over the first dimension."""
        ...

    @abstractmethod
    def __getitem__(self, key: Any, /) -> Self | Scalar:
        """Return an element/slice of the table."""
        ...

    @abstractmethod
    def all(self) -> Self | bool:
        """Return True if all elements are True."""
        ...

    @abstractmethod
    def any(self) -> Self | bool:
        """Return True if any element is True."""
        ...

    # region arithmetic operations -----------------------------------------------------
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

    # endregion arithmetic operations --------------------------------------------------


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


ArrayType = TypeVar("ArrayType", bound=ArrayKind)
TableType = TypeVar("TableType", bound=TableKind)
NumericalArrayType = TypeVar("NumericalArrayType", bound=NumericalArray)
MutableArrayType = TypeVar("MutableArrayType", bound=MutableArray)
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
class SupportsGetItem(Protocol[K_contra, V_co]):
    """Protocol for objects that support `__getitem__`."""

    def __getitem__(self, key: K_contra, /) -> V_co: ...


@runtime_checkable
class SupportsKeysAndGetItem(Protocol[K, V_co]):
    """Protocol for objects that support `__getitem__` and `keys`."""

    def keys(self) -> Iterable[K]: ...
    def __getitem__(self, key: K, /) -> V_co: ...


@runtime_checkable
class SupportsLenAndGetItem(Protocol[V_co]):
    """Protocol for objects that support integer based `__getitem__` and `__len__`."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: int, /) -> V_co: ...


class SupportsKwargsType(type(Protocol)):  # type: ignore[misc]
    """Metaclass for `SupportsKwargs`."""

    def __instancecheck__(self, other: object, /) -> bool:
        return isinstance(other, SupportsKeysAndGetItem) and all(
            isinstance(key, str) for key in other.keys()
        )

    def __subclasscheck__(self, other: type, /) -> bool:
        raise NotImplementedError("Cannot check whether a class is a SupportsKwargs.")


@runtime_checkable
class SupportsKwargs(Protocol[V_co], metaclass=SupportsKwargsType):
    """Protocol for objects that support `**kwargs`."""

    def keys(self) -> Iterable[str]: ...
    def __getitem__(self, key: str, /) -> V_co: ...


@runtime_checkable
class SequenceProtocol(Protocol[T_co]):
    """Protocol version of `collections.abc.Sequence`.

    Note:
        We intentionally exclude `Reversible`, since `tuple` fakes this:
        `tuple` has not attribute `__reversed__`, rather it used the
        `Sequence.register(tuple)` to artificially become a nominal subtype.

    References:
        - https://github.com/python/typeshed/blob/main/stdlib/typing.pyi
        - https://github.com/python/cpython/blob/main/Lib/_collections_abc.py
    """

    @abstractmethod
    def __len__(self) -> int: ...
    @overload
    @abstractmethod
    def __getitem__(self, index: int, /) -> T_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, index: slice, /) -> "SequenceProtocol[T_co]": ...

    # Mixin methods
    # TODO: implement mixin methods
    # def __reversed__(self) -> Iterator[T_co]: ...  # NOTE: intentionally excluded
    def __iter__(self) -> Iterator[T_co]: ...
    def __contains__(self, value: object, /) -> bool: ...
    def index(self, value: Any, start: int = 0, stop: int = ..., /) -> int: ...
    def count(self, value: Any, /) -> int: ...


@runtime_checkable
class MutableSequenceProtocol(SequenceProtocol[T], Protocol[T]):
    """Protocol version of `collections.abc.MutableSequence`."""

    @overload
    @abstractmethod
    def __getitem__(self, index: int, /) -> T: ...
    @overload
    @abstractmethod
    def __getitem__(self, index: slice, /) -> "MutableSequenceProtocol[T]": ...
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

    # Mixin methods
    # TODO: Implement mixin methods
    def append(self, value: T, /) -> None: ...
    def clear(self, /) -> None: ...
    def extend(self, values: Iterable[T], /) -> None: ...
    def reverse(self, /) -> None: ...
    def pop(self, index: int = -1, /) -> T: ...
    def remove(self, value: T, /) -> None: ...
    def __iadd__(self, values: Iterable[T], /) -> Self: ...


@runtime_checkable
class MappingProtocol(Collection[K], Protocol[K, V_co]):
    """Protocol version of `collections.abc.Mapping`."""

    # FIXME: implement mixin methods

    @abstractmethod
    def __getitem__(self, __key: K, /) -> V_co: ...

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
    def __setitem__(self, __key: K, __value: V, /) -> None: ...
    @abstractmethod
    def __delitem__(self, __key: K, /) -> None: ...

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
        self: "MutableMappingProtocol[K, T | None]", __key: K, __default: None = ...
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
        if (typing.NamedTuple in get_original_bases(other)) or (
            typing_extensions.NamedTuple in get_original_bases(other)
        ):
            return True
        return NotImplemented


@overload
def is_dataclass(obj: type, /) -> TypeGuard[type[Dataclass]]: ...
@overload
def is_dataclass(obj: object, /) -> TypeGuard[Dataclass]: ...
def is_dataclass(obj, /):  # pyright: ignore
    r"""Check if the object is a dataclass."""
    if isinstance(obj, type):
        return issubclass(obj, Dataclass)  # type: ignore[misc]
    return issubclass(type(obj), Dataclass)  # type: ignore[misc]


@overload
def is_namedtuple(obj: type, /) -> TypeGuard[type[NTuple]]: ...
@overload
def is_namedtuple(obj: object, /) -> TypeGuard[NTuple]: ...
def is_namedtuple(obj, /):  # pyright: ignore
    """Check if the object is a namedtuple."""
    if isinstance(obj, type):
        return issubclass(obj, NTuple)  # type: ignore[misc]
    return issubclass(type(obj), NTuple)  # type: ignore[misc]


# NOTE: TypedDict not supported, since instances of type-dicts forget their parent class.
#  and are just plain dicts.

# endregion generic factory-protocols --------------------------------------------------
