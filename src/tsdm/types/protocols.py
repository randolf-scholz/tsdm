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
    # Mixins
    "SupportsArray",
    "SupportsArrayUfunc",
    "SupportsDataframe",
    "SupportsDevice",
    "SupportsDtype",
    "SupportsItem",
    "SupportsNdim",
    "SupportsShape",
    # Scalars
    "BaseScalar",
    "BooleanScalar",
    "OrderedScalar",
    "AdditiveScalar",
    # Arrays
    "SeriesKind",
    "TableKind",
    "ArrayKind",
    "NumericalArray",
    "NumericalSeries",
    "NumericalTensor",
    "MutableArray",
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
from types import GenericAlias, get_original_bases
from typing import (
    Any,
    ClassVar,
    Literal,
    Optional,
    Protocol,
    Self,
    SupportsIndex,
    overload,
    runtime_checkable,
)

import numpy as np
import typing_extensions
from numpy.typing import NDArray
from typing_extensions import TypeIs

from tsdm.types.aliases import MultiIndexer

# region io protocols ------------------------------------------------------------------


@runtime_checkable
class BaseBuffer(Protocol):
    r"""Base class for ReadBuffer and WriteBuffer."""

    # REF: WriteBuffer from https://github.com/pandas-dev/pandas/blob/main/pandas/_typing.py
    # REF: SupportsWrite from https://github.com/python/typeshed/blob/main/stdlib/_typeshed/__init__.pyi
    # REF: IOBase from https://github.com/python/typeshed/blob/main/stdlib/io.pyi
    # REF: IO from https://github.com/python/typeshed/blob/main/stdlib/typing.pyi
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
class BaseScalar(Protocol):
    r"""Protocol for scalars."""

    def __eq__(self, other: Self, /) -> "BooleanScalar": ...  # type: ignore[override]
    def __ne__(self, other: Self, /) -> "BooleanScalar": ...  # type: ignore[override]


@runtime_checkable
class OrderedScalar(BaseScalar, Protocol):
    r"""Protocol for ordered scalars.

    Examples:
        - `bool`, `int`, `float`, `datetime`, `timedelta`

    Counter-Examples:
        - `complex` (not ordered)
    """

    def __ge__(self, other: Self, /) -> "BooleanScalar": ...
    def __gt__(self, other: Self, /) -> "BooleanScalar": ...
    def __le__(self, other: Self, /) -> "BooleanScalar": ...
    def __lt__(self, other: Self, /) -> "BooleanScalar": ...


@runtime_checkable
class BooleanScalar(OrderedScalar, Protocol):
    r"""Protocol for boolean scalars."""

    # unary operations
    def __bool__(self) -> bool: ...

    # NOTE: __invert__ is not included, as ~True = -2, which is not a boolean.
    # def __invert__(self) -> Self: ...

    # binary operations
    def __and__(self, other: Self, /) -> Self: ...
    def __or__(self, other: Self, /) -> Self: ...
    def __xor__(self, other: Self, /) -> Self: ...
    def __ror__(self, other: Self, /) -> Self: ...
    def __rand__(self, other: Self, /) -> Self: ...
    def __rxor__(self, other: Self, /) -> Self: ...


@runtime_checkable
class AdditiveScalar(BaseScalar, Protocol):
    r"""Protocol for scalars that support addition and subtraction.

    Examples:
        - `int`, `float`, `complex`, `timedelta`

    Counter-Examples:
        - `bool` (does not support __pos__ and __neg__)
        - `datetime.datetime` (does not support self-addition)
    """

    # unary operations
    # NOTE: __abs__ disabled due to complex numbers
    # def __abs__(self) -> Self: ...
    def __neg__(self) -> Self: ...
    def __pos__(self) -> Self: ...

    # binary operations
    def __add__(self, other: Self, /) -> Self: ...
    def __radd__(self, other: Self, /) -> Self: ...
    def __sub__(self, other: Self, /) -> Self: ...
    def __rsub__(self, other: Self, /) -> Self: ...


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
    def __add__(self, other: Self | tuple, /) -> "ShapeLike": ...
    def __eq__(self, other: object, /) -> bool: ...
    def __ne__(self, other: object, /) -> bool: ...
    def __lt__(self, other: Self | tuple, /) -> bool: ...
    def __le__(self, other: Self | tuple, /) -> bool: ...


@runtime_checkable
class SupportsDevice(Protocol):
    r"""Protocol for objects that support `device`."""

    @property
    @abstractmethod
    def device(self) -> Any:
        r"""Return the device of the tensor."""
        ...


@runtime_checkable
class SupportsDtype(Protocol):
    r"""We just test for dtype, since e.g. tf.Tensor does not have ndim.

    Examples:
        - `numpy.ndarray`
        - `pandas.Series`
        - `polars.Series`
        - `torch.Tensor`

    Counter-Examples:
        - `pandas.DataFrame`
        - `polars.DataFrame`
        - `pyarrow.Array`
    """

    @property
    @abstractmethod
    def dtype(self) -> Any:
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
        r"""Yield the shape of the array."""
        ...


@runtime_checkable
class SupportsArray(Protocol):
    r"""Protocol for objects that support `__array__`.

    See: https://numpy.org/doc/stable/reference/c-api/array.html
    """

    @abstractmethod
    def __array__(self) -> NDArray[np.object_]:
        r"""Return the array of the tensor."""
        ...


@runtime_checkable
class SupportsArrayUfunc(SupportsArray, Protocol):
    r"""Protocol for objects that support `__array_ufunc__`.

    Notably, numpy functions like `numpy.exp` can be directly applied to such objects.
    The main example are `pandas.Series` and `pandas.DataFrame`.

    Examples:
        - `numpy.ndarray`
        - `pandas.Series`
        - `pandas.DataFrame`
        - `polars.Series`

    Counter-Examples:
        - `polars.DataFrame`
        - `pyarrow.Array`
        - `pyarrow.Table`
        - `torch.Tensor`

    References:
        - https://numpy.org/doc/stable/reference/ufuncs.html
    """

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "at"],
        *inputs: Any,
        **kwargs: Any,
    ) -> Self:
        r"""Return the array resulting from applying the ufunc."""
        ...


@runtime_checkable
class SupportsDataframe(Protocol):
    r"""Protocol for objects that support `__dataframe__`.

    See: https://data-apis.org/dataframe-protocol/latest/index.html
    """

    @abstractmethod
    def __dataframe__(self) -> Any:
        r"""Return the dataframe of the tensor."""
        ...


@runtime_checkable
class SupportsItem[T](Protocol):  # +T
    r"""Protocol for objects that support `.item()`."""

    @abstractmethod
    def item(self) -> T:
        r"""Return the scalar value the tensor if it only has a single element.

        If the tensor has more than one element, raise an error.
        """
        ...


@runtime_checkable
class ArrayKind[Scalar](Protocol):
    r"""An n-dimensional array of a single homogeneous data type.

    Examples:
        - `numpy.ndarray`
        - `pandas.DataFrame`
        - `pandas.Series`
        - `polars.DataFrame`
        - `polars.Series`
        - `pyarrow.Array`
        - `pyarrow.Table`
        - `torch.Tensor`

    References:
        - https://docs.python.org/3/c-api/buffer.html
        - https://numpy.org/doc/stable/reference/arrays.interface.html
        - https://numpy.org/devdocs/user/basics.interoperability.html
    """

    # NOTE: This is a highly cut down version, to support the bare minimum.

    @property
    def shape(self) -> tuple[int, ...]: ...

    def __array__(self) -> NDArray: ...
    def __len__(self) -> int: ...

    # comparisons (element-wise)
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


@runtime_checkable
class SeriesKind[Scalar](Protocol):
    r"""A 1d-array of homogeneous data type.

    Examples:
        - `pandas.Index`
        - `pandas.Series`
        - `polars.Series`
        - `pyarrow.Array`

    Counter-Examples:
        - `numpy.ndarray`     lacks `equals`
        - `pandas.DataFrame`  lacks `equals`
        - `polars.DataFrame`  lacks `equals`
        - `pyarrow.Table`     lacks `equals`
        - `torch.Tensor`      lacks `equals`

    References:
        - https://numpy.org/devdocs/user/basics.interoperability.html
    """

    # NOTE: The following methods differ between backends:
    #  - diff: gives discrete differences for polars and pandas, but not for pyarrow
    #  - value_counts: polars returns a DataFrame, pandas a Series, pyarrow a StructArray
    # NOTE: We do not include to_numpy(), as this is covered by __array__.

    def __array__(self) -> NDArray: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Scalar]: ...
    @overload
    def __getitem__(self, key: int, /) -> Scalar: ...
    @overload
    def __getitem__(self, key: slice, /) -> Self: ...

    def unique(self) -> Self:
        r"""Return the unique elements of the series."""
        ...

    def equals(self, other: Self) -> bool:
        r"""Check if the series is equal to another series."""
        ...

    def value_counts(self) -> SupportsArray:
        r"""For each unique value holds the number of counts."""
        ...

    # comparisons (element-wise)
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


@runtime_checkable
class TableKind(Protocol):
    r"""A 2d column-oriented array with heterogenous data types.

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

    References:
        - https://numpy.org/devdocs/user/basics.interoperability.html
        - https://data-apis.org/dataframe-protocol/latest/index.html
    """

    # NOTE: The following methods differ between backends:
    #  - __iter__: yields columns for polars and pandas, but rows for pyarrow
    #  - take: not supported by polars
    #  - columns: pandas and polars return column names, pyarrow returns list of Arrays
    #  - drop: polars currently doing signature change
    #  - filter: pandas goes over columns, polars over rows

    @property
    def shape(self) -> tuple[int, int]: ...

    def __array__(self) -> NDArray[np.object_]: ...
    def __dataframe__(self, *, allow_copy: bool = True) -> object: ...
    def __len__(self) -> int: ...

    @overload
    def __getitem__(self, key: str, /) -> SeriesKind: ...
    @overload
    def __getitem__(self, key: slice, /) -> Self: ...

    def equals(self, other: Self, /) -> bool:
        r"""Check if the table is equal to another table."""
        ...

    # comparisons (element-wise)
    # equality ==
    def __eq__(self, other: Self | object, /) -> Self: ...  # type: ignore[override]
    # inequality !=
    def __ne__(self, other: Self | object, /) -> Self: ...  # type: ignore[override]
    # less than or equal <=
    def __le__(self, other: Self | object, /) -> Self: ...
    # greater than or equal >=
    def __ge__(self, other: Self | object, /) -> Self: ...
    # less than <
    def __lt__(self, other: Self | object, /) -> Self: ...
    # greater than >
    def __gt__(self, other: Self | object, /) -> Self: ...


@runtime_checkable
class NumericalArray[Scalar](ArrayKind[Scalar], Protocol):
    r"""Subclass of `ArrayKind` that supports numerical operations.

    Examples:
        - `numpy.ndarray`
        - `pandas.Index`
        - `pandas.Series`
        - `pandas.DataFrame`
        - `polars.Series`
        - `torch.Tensor`

    Counter-Examples:
        - `polars.DataFrame`  (does not support basic arithmetic)
        - `pyarrow.Array`  (does not support basic arithmetic)
        - `pyarrow.Table`  (does not support basic arithmetic)

    References:
        - This is a weak version of the Array API:
          https://data-apis.org/array-api/latest/API_specification/array_object.html
        - https://data-apis.org/dataframe-api/draft/index.html
        - https://numpy.org/devdocs/user/basics.interoperability.html
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    def __array__(self) -> NDArray: ...
    def __len__(self) -> int: ...

    # NOTE: This is weakly typed since it returns different things on different objects.
    def __getitem__(self, key: Any, /) -> "Self | NumericalSeries | Scalar": ...

    def all(self) -> Self | BooleanScalar:
        r"""Return True if all elements are True."""
        ...

    def any(self) -> Self | BooleanScalar:
        r"""Return True if any element is True."""
        ...

    def min(self) -> Scalar:
        r"""Return the minimum value."""
        ...

    def max(self) -> Scalar:
        r"""Return the maximum value."""
        ...

    # FIXME: https://github.com/python/typing/discussions/1782
    @overload
    def round(self) -> Self: ...
    @overload
    def round(self, *, decimals: int) -> Self: ...
    def round(self, *, decimals: int = 0) -> Self:
        r"""Round the array to the given number of decimals."""
        ...

    # region arithmetic operations -----------------------------------------------------
    # unary operations
    # absolute value abs()
    def __abs__(self) -> Self: ...
    # bitwise NOT ~
    def __invert__(self) -> Self: ...
    # negation -
    def __neg__(self) -> Self: ...
    # positive +
    def __pos__(self) -> Self: ...

    # comparisons (element-wise)
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
    # NOTE: polars does not support complex data types!
    def __pow__(self, exponent: Self | float, /) -> Self: ...
    def __rpow__(self, base: Self | float, /) -> Self: ...
    # modulo %
    def __mod__(self, other: Self | Scalar, /) -> Self: ...
    def __rmod__(self, other: Self | Scalar, /) -> Self: ...

    # matrix multiplication @
    # def __matmul__(self, other: Self, /) -> Self: ...
    # def __rmatmul__(self, other: Self, /) -> Self: ...

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
    # left shift <<
    # def __lshift__(self, other: Self | Scalar, /) -> Self: ...
    # def __rlshift__(self, other: Self | Scalar, /) -> Self: ...
    # right shift >>
    # def __rshift__(self, other: Self | Scalar, /) -> Self: ...
    # def __rrshift__(self, other: Self | Scalar, /) -> Self: ...

    # endregion arithmetic operations --------------------------------------------------


class NumericalSeries[Scalar](NumericalArray[Scalar], Protocol):
    r"""Protocol for numerical series.

    Series are per definition one dimensional, and have a unique data type.
    Notably, this differs with respect to `NumericalTensor` by not supporting tuple-indexing.
    Moreover, its Iterator and `__getitem__(int)` are guaranteed to return scalars.

    Note:
        Multidimensional Tensors are by definition Series of Tensors.
        For instance, a 3-dimensional numpy array is a `NumericalSeries[NDArray]`.

    Examples:
        - `numpy.ndarray`
        - `pandas.Index`
        - `pandas.Series`
        - `polars.Series`
        - `torch.Tensor`

    Counter-Examples:
        - `pandas.DataFrame`
        - `polars.DataFrame`
    """

    @property
    def dtype(self) -> Any: ...

    def __iter__(self) -> Iterator[Scalar]: ...

    # fmt: off
    @overload  # depending on Tensor Rank, can return Scalar or Tensor
    def __getitem__(self, key: int, /) -> Scalar: ...
    @overload
    def __getitem__(self, key: slice | range | list[int], /) -> Self: ...
    # fmt: on

    def item(self) -> Scalar:
        r"""Return the scalar value the tensor if it only has a single element.

        Otherwise, raises `ValueError`.
        """
        ...


class NumericalTensor[Scalar](NumericalSeries[Scalar], Protocol):
    r"""Protocol for numerical tensors.

    Compared to `NumericalSeries`, tensors *can* have multiple dimensions, and
    must support more Indexing operations, In particular `...` (Ellipsis), and
    tuples of ints and/or slices.

    Examples:
        - `numpy.ndarray`
        - `pandas.Index`
        - `pandas.Series`
        - `torch.Tensor`

    Counter-Examples:
        - `polars.Series`    (cannot be indexed with Ellipsis and tuple)
    """

    @property
    def dtype(self) -> Any: ...
    @property
    def ndim(self) -> int: ...

    # fmt: off
    @overload  # depending on Tensor Rank, can return Scalar or Tensor
    def __getitem__(self, key: int | tuple[int, ...], /) -> Scalar | Self: ...  # type: ignore[overload-overlap]
    @overload
    def __getitem__(self, key: Self | MultiIndexer, /) -> Self: ...
    # fmt: on

    def item(self) -> Scalar:
        r"""Return the scalar value the tensor if it only has a single element.

        Otherwise, raises `ValueError`.
        """
        ...

    def argmin(self) -> int | Self:
        r"""Return the index of the minimum value."""
        ...

    def argmax(self) -> int | Self:
        r"""Return the index of the maximum value."""
        ...

    def ravel(self) -> Self:
        r"""Return a flattened version of the tensor."""
        ...


@runtime_checkable
class MutableArray[Scalar](NumericalArray[Scalar], Protocol):
    r"""Subclass of `Array` that supports inplace operations.

    Examples:
        - `numpy.ndarray`
        - `pandas.DataFrame`
        - `pandas.Series`
        - `torch.Tensor`

    Counter-Examples:
        - `pandas.Index`     (does not support inplace operations)
        - `polars.DataFrame` (does not support inplace operations)
        - `polars.Series`    (does not support inplace operations)
        - `pyarrow.Array`    (does not support inplace operations)
        - `pyarrow.Table`    (does not support inplace operations)
    """

    def take(self, indices: Any, /) -> Self:
        r"""Select elements from the array by index."""
        ...

    # NOTE: The following operations are excluded:
    #  - __imatmul__: because it potentially changes the shape of the array.

    # matrix multiplication @
    def __matmul__(self, other: Self, /) -> Self: ...
    def __rmatmul__(self, other: Self, /) -> Self: ...

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
    def __ipow__(self, power: Self | float, /) -> Self: ...
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

    # # inplace bitwise operations
    # # left shift =<<
    # def __ilshift__(self, other: Self | Scalar, /) -> Self: ...
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


class _SupportsKwargsMeta(type(Protocol)):  # type: ignore[misc]
    r"""Metaclass for `SupportsKwargs`."""

    def __instancecheck__(cls, other: object, /) -> TypeIs["SupportsKwargs"]:
        return isinstance(other, SupportsKeysAndGetItem) and all(
            isinstance(key, str)
            for key in other.keys()  # noqa: SIM118
        )

    def __subclasscheck__(cls, other: type, /) -> TypeIs[type["SupportsKwargs"]]:
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
    def isdisjoint(self, other: Iterable, /) -> bool: ...


class _ArrayMeta(type(Protocol)):  # type: ignore[misc]
    def __subclasscheck__(cls, other: type) -> TypeIs[type["Array"]]:
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

    @overload
    def get(self, key: K, /) -> Optional[V]: ...
    @overload
    def get[T](self, key: K, /, default: V | T) -> V | T: ...
    def get(self, key, /, default=None):
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
    @overload
    def pop(self, key: K, /) -> V: ...
    @overload
    def pop(self, key: K, /, default: V) -> V: ...
    @overload
    def pop[T](self, key: K, /, default: T) -> V | T: ...
    def popitem(self) -> tuple[K, V]: ...
    @overload
    def setdefault[T](
        self: "MutMap[K, T | None]", key: K, /, default: None = ...
    ) -> T | None: ...
    @overload
    def setdefault(self, key: K, /, default: V) -> V: ...
    @overload
    def update(self, m: SupportsKeysAndGetItem[K, V], /, **kwargs: V) -> None: ...
    @overload
    def update(self, m: Iterable[tuple[K, V]], /, **kwargs: V) -> None: ...
    @overload
    def update(self, **kwargs: V) -> None: ...


# endregion stdlib protocols -----------------------------------------------------------


# region generic factory-protocols -----------------------------------------------------


@runtime_checkable
class Dataclass(Protocol):
    r"""Protocol for anonymous dataclasses.

    Similar to `DataClassInstance` from typeshed, but allows isinstance and issubclass.
    """

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]] = {}
    r"""The fields of the dataclass."""

    @classmethod
    def __subclasshook__(cls, other: type, /) -> bool:
        r"""Cf https://github.com/python/cpython/issues/106363."""
        fields = getattr(other, "__dataclass_fields__", None)
        return isinstance(fields, dict)


@runtime_checkable  # FIXME: Use TypeVarTuple
class NTuple[T](Protocol):  # +T
    r"""Protocol for anonymous namedtuple.

    References:
        - https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi
    """

    # FIXME: Problems if denoted as  tuple[str, ...]
    #    Should be tuple[*(str for T in Ts)] (not tuple[str, ...])
    #   see: https://github.com/python/typing/issues/1216
    #   see: https://github.com/python/typing/issues/1273
    _fields: tuple
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

    @classmethod
    def __subclasshook__(cls, other: type, /) -> bool:
        r"""Cf https://github.com/python/cpython/issues/106363."""
        bases = get_original_bases(other)
        return (typing.NamedTuple in bases) or (typing_extensions.NamedTuple in bases)


class _SlottedMeta(type(Protocol)):
    r"""Metaclass for `Slotted`.

    FIXME: https://github.com/python/cpython/issues/112319
    This issue will make the need for metaclass obsolete.
    """

    def __instancecheck__(self, instance: object) -> TypeIs["Slotted"]:
        slots = getattr(instance, "__slots__", None)
        return isinstance(slots, str | Iterable)

    def __subclasscheck__(cls, other: type, /) -> TypeIs[type["Slotted"]]:
        slots = getattr(other, "__slots__", None)
        return isinstance(slots, str | Iterable)


@runtime_checkable
class Slotted(Protocol, metaclass=_SlottedMeta):
    r"""Protocol for objects that are slotted."""

    __slots__: tuple[str, ...] = ()


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
