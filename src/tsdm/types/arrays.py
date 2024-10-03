r"""Array protocol types for the numerical backend.

Here is a summary, based on the library versions:

- `numpy`:
- `pandas`:
- `polars`:
- `torch`:
- `pyarrow`:

|                    | np.ndarray | torch.Tensor | pd.Index | pd.EA | pd.Series  | pd.DataFrame | pl.Series | pl.DataFrame | pa.Array | pa.Table |
|--------------------|------------|--------------|----------|-------|------------|--------------|-----------|--------------|----------|----------|
| dimensionality     | N          | N            | 1        | 1     | 1          | 2            | 1         | 2            | 1        | 2        |
| comparisons        | ✅          | ✅            | ✅        | ✅     | ✅          | ✅            | ✅         | ✅            | ❌        | ❌        |
| Arithmetic         | ✅          | ✅            | ✅        | ✅     | ✅          | ✅            | ✅         | ❌            | ❌        | ❌        |
| inplace arith.     | ✅          | ✅            | ❌        | ❌     | ✅          | ✅            | ❌         | ❌            | ❌        | ❌        |
| `__array__`        | ✅          | ✅            | ✅        | ✅     | ✅          | ✅            | ✅         | ✅            | ✅        | ✅        |
| `__array_ufunc__`  | ✅          | ❌            | ✅        | ✅     | ✅          | ✅            | ❌         | ❌            | ❌        | ❌        |
| `__dataframe__`    | ❌          | ❌            | ❌        | ❌     | ❌          | ✅            | ❌         | ✅            | ❌        | ✅        |
| `.shape`           | ✅          | ✅            | ✅        | ✅     | ✅          | ✅            | ✅         | ✅            | ❌        | ✅        |
| `.dtype`           | ✅          | ✅            | ✅        | ✅     | ✅          | ❌            | ✅         | ❌            | ❌        | ❌        |
| `.ndim`            | ✅          | ✅            | ✅        | ✅     | ✅          | ✅            | ❌         | ❌            | ❌        | ❌        |
| `.device`          | ✅          | ✅            | ❌        | ❌     | ❌          | ❌            | ❌         | ❌            | ❌        | ❌        |
| `item()`           | ✅          | ✅            | ✅        | ❌     | ✅          | ❌            | ✅         | ✅            | ❌        | ❌        |
| `__matmul__()`     | ✅          | ✅            | ❌        | ❌     | ✅          | ✅            | ✅         | ❌            | ❌        | ❌        |
| `__len__()`        | ✅          | ✅            | ✅        | ✅     | ✅          | ✅            | ✅         | ✅            | ✅        | ✅        |
| `__iter__()`       | ✅          | ✅            | ✅        | ✅     | ✅          | ✅            | ✅         | ✅            | ✅        | ❌        |
| `iter() -> Self`   | ✅          | ✅            | ❌        | ❌     | ❌          | ❌            | ❌         | ❌            | ❌        | ❌        |
| `iter() -> ?`      | ROW        | ROW          | ROW      | ROW   | ROW        |              | COL NAME  | COL          | ROW      | COL      |
| `__getitem__[int]` | ✅          | ✅            | ✅        | ✅     | ⚡ UNSAFE ⚡ | ❌            | ✅         | ROW          | ROW      | COL      |
| `__getitem__[str]` |            |              |          |       |            |              |           | COL          | ❌        | COL      |

From this table, we are inclined to derive several protocols:

Warning:
    `polars` and `pyarrow` are strict by allowing columns to only be indexed by strings, and rows by integers.
    This creates type-safety. `pandas` on the other hand does not have this safety.
    For example, `series[0]` will select the first row if `0` is not in the index,
    but if `0` is in the index, it will select the row(s) whose index is equal to `0`.
"""  # noqa: E501, W505

__all__ = [
    "SupportsDevice",
    "SupportsDtype",
    "SupportsNdim",
    "SupportsShape",
    "SupportsArray",
    "SupportsRound",
    "SupportsArrayUfunc",
    "SupportsDataFrame",
    "SupportsItem",
    "SupportsArithmetic",
    "SupportsInplaceArithmetic",
    "SupportsComparison",
    "SupportsMatmul",
    "ArrayKind",
    "SeriesKind",
    "TableKind",
    "NumericalArray",
    "NumericalSeries",
    "NumericalTensor",
    "MutableArray",
]


from abc import abstractmethod
from collections.abc import Iterator
from typing import Any, Literal, Protocol, Self, overload, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from tsdm.types.aliases import MultiIndexer
from tsdm.types.scalars import BaseScalar, BooleanScalar

# region helper protocols --------------------------------------------------------------


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
class SupportsRound(Protocol):
    r"""Protocol for objects that support `round`."""

    # FIXME: https://github.com/python/typing/discussions/1782
    @overload
    def round(self) -> Self: ...
    @overload
    def round(self, *, decimals: int) -> Self: ...
    def round(self, *, decimals: int = 0) -> Self:
        r"""Round to the given number of decimals."""
        ...


@runtime_checkable
class SupportsItem[Scalar](Protocol):  # +T
    r"""Protocol for objects that support `.item()`."""

    @abstractmethod
    def item(self) -> Scalar:
        r"""Return the scalar value the tensor if it only has a single element.

        If the tensor has more than one element, raise an error.
        """
        ...


# endregion helper protocols -----------------------------------------------------------


# region mixin protocols ---------------------------------------------------------------
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
class SupportsDataFrame(Protocol):
    r"""Protocol for objects that support `__dataframe__`.

    See: https://data-apis.org/dataframe-protocol/latest/index.html
    """

    @abstractmethod
    def __dataframe__(self) -> Any:
        r"""Return the dataframe of the tensor."""
        ...


@runtime_checkable
class SupportsArrayUfunc(SupportsArray, Protocol):
    r"""Protocol for objects that support `__array_ufunc__`.

    Notably, numpy functions like `numpy.exp` can be directly applied to such objects.
    The main example are `pandas.Series` and `pandas.DataFrame`.

    Examples:
        - `numpy.ndarray`
        - `pandas.Index`
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
class SupportsArithmetic[Scalar](Protocol):
    r"""Mixin Protocol for arithmetic operations.

    Note:
        Excludes matrix multiplication as well as shift operations.
    """

    # unary operations
    # absolute value abs()
    def __abs__(self) -> Self: ...
    # bitwise NOT ~
    def __invert__(self) -> Self: ...
    # negation -
    def __neg__(self) -> Self: ...
    # positive +
    def __pos__(self) -> Self: ...

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

    # matrix multiplication @
    # def __matmul__(self, other: Self, /) -> Self: ...
    # def __rmatmul__(self, other: Self, /) -> Self: ...

    # bitwise operators
    # left shift <<
    # def __lshift__(self, other: Self | Scalar, /) -> Self: ...
    # def __rlshift__(self, other: Self | Scalar, /) -> Self: ...
    # right shift >>
    # def __rshift__(self, other: Self | Scalar, /) -> Self: ...
    # def __rrshift__(self, other: Self | Scalar, /) -> Self: ...


@runtime_checkable
class SupportsInplaceArithmetic[Scalar](Protocol):
    """Mixin Protocol for inplace arithmetic operations.

    Note:
        Excludes inplace matrix multiplication as well as inplace shift operations.
    """

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


@runtime_checkable
class SupportsComparison[Scalar](Protocol):
    r"""Mixin Protocol for comparison operations."""

    # comparisons (element-wise)
    # equality ==
    def __eq__(self, other: object, /) -> Self: ...  # type: ignore[override]
    # inequality !=
    def __ne__(self, other: object, /) -> Self: ...  # type: ignore[override]
    # less than or equal <=
    def __le__(self, other: Self | Scalar, /) -> Self: ...
    # greater than or equal >=
    def __ge__(self, other: Self | Scalar, /) -> Self: ...
    # less than <
    def __lt__(self, other: Self | Scalar, /) -> Self: ...
    # greater than >
    def __gt__(self, other: Self | Scalar, /) -> Self: ...


@runtime_checkable
class SupportsMatmul(Protocol):
    r"""Mixin Protocol for matrix multiplication operations."""

    # matrix multiplication @
    def __matmul__(self, other: Self, /) -> Self: ...
    def __rmatmul__(self, other: Self, /) -> Self: ...


# endregion mixin protocols ------------------------------------------------------------


@runtime_checkable
class ArrayKind[Scalar](Protocol):
    r"""An n-dimensional array of a single homogeneous data type.

    Examples:
        - `numpy.ndarray`
        - `pandas.DataFrame`
        - `pandas.Series`
        - `pandas.extensions.ExtensionArray`
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
    def __eq__(self, other: object, /) -> Self: ...  # type: ignore[override]
    # inequality !=
    def __ne__(self, other: object, /) -> Self: ...  # type: ignore[override]
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
        - `pandas.extensions.ExtensionArray`
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
    def __eq__(self, other: object, /) -> Self: ...  # type: ignore[override]
    # inequality !=
    def __ne__(self, other: object, /) -> Self: ...  # type: ignore[override]
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
        - `pandas.extensions.ExtensionArray`  lacks `__dataframe__`
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
    def __eq__(self, other: object, /) -> Self: ...  # type: ignore[override]
    # inequality !=
    def __ne__(self, other: object, /) -> Self: ...  # type: ignore[override]
    # less than or equal <=
    def __le__(self, other: Self | object, /) -> Self: ...
    # greater than or equal >=
    def __ge__(self, other: Self | object, /) -> Self: ...
    # less than <
    def __lt__(self, other: Self | object, /) -> Self: ...
    # greater than >
    def __gt__(self, other: Self | object, /) -> Self: ...


@runtime_checkable
class NumericalArray[Scalar](ArrayKind[Scalar], Protocol):  # -Scalar
    r"""Subclass of `ArrayKind` that supports numerical operations.

    Examples:
        - `numpy.ndarray`
        - `pandas.Index`
        - `pandas.Series`
        - `pandas.extensions.ExtensionArray`
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
    def __getitem__(self, key: Any, /) -> "Self | NumericalSeries | BaseScalar": ...

    def all(self) -> Self | BooleanScalar:
        r"""Return True if all elements are True."""
        ...

    def any(self) -> Self | BooleanScalar:
        r"""Return True if any element is True."""
        ...

    def min(self) -> Self | Scalar:
        r"""Return the minimum value."""
        ...

    def max(self) -> Self | Scalar:
        r"""Return the maximum value."""
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
    def __eq__(self, other: object, /) -> Self: ...  # type: ignore[override]
    # inequality !=
    def __ne__(self, other: object, /) -> Self: ...  # type: ignore[override]
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
        - `pandas.extensions.ExtensionArray`
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
    def __getitem__(self, key: slice | range | list[int] | Self, /) -> Self: ...
    # fmt: on

    def item(self) -> Scalar:
        r"""Return the scalar value the tensor if it only has a single element.

        Otherwise, raises `ValueError`.
        """
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
        - `pandas.extensions.ExtensionArray`  (does not support inplace operations)
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


class NumericalTensor[Scalar](NumericalArray[Scalar], Protocol):
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
        - `polars.DataFrame` (cannot be indexed with Ellipsis and tuple)
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

    # FIXME: https://github.com/python/typing/discussions/1782
    @overload
    def round(self) -> Self: ...
    @overload
    def round(self, *, decimals: int) -> Self: ...
    def round(self, *, decimals: int = 0) -> Self:
        r"""Round the array to the given number of decimals."""
        ...
