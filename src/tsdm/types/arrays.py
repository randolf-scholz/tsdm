r"""Array protocol types for the numerical backend.

Here is a summary, based on the library versions:

- `numpy`:
- `pandas`:
- `polars`:
- `torch`:
- `pyarrow`:

|                             | np.ndarray | torch.Tensor | pd.Index | pd.EA | pd.Series       | pl.Series | pa.Array | pd.DataFrame | pl.DataFrame | pa.Table |
|-----------------------------|------------|--------------|----------|-------|-----------------|-----------|----------|--------------|--------------|----------|
| dimensionality              | N          | N            | 1        | 1     | 1               | 1         | 1        | 2            | 2            | 2        |
| comparisons                 | ✅          | ✅            | ✅        | ✅     | ✅               | ✅         | ❌        | ✅            | ✅            | ❌        |
| arithmetic                  | ✅          | ✅            | ✅        | ✅     | ✅               | ✅         | ❌        | ✅            | ❌            | ❌        |
| mutation                    | ✅          | ✅            | ❌        | ❌     | ✅               | ❌         | ❌        | ✅            | ❌            | ❌        |
| `__array__`                 | ✅          | ✅            | ✅        | ✅     | ✅               | ✅         | ✅        | ✅            | ✅            | ✅        |
| `__array_ufunc__`           | ✅          | ❌            | ✅        | ✅     | ✅               | ❌         | ❌        | ✅            | ❌            | ❌        |
| `__dataframe__`             | ❌          | ❌            | ❌        | ❌     | ❌               | ❌         | ❌        | ✅            | ✅            | ✅        |
| `.shape`                    | ✅          | ✅            | ✅        | ✅     | ✅               | ✅         | ❌        | ✅            | ✅            | ✅        |
| `.dtype`                    | ✅          | ✅            | ✅        | ✅     | ✅               | ✅         | ❌        | ❌            | ❌            | ❌        |
| `.ndim`                     | ✅          | ✅            | ✅        | ✅     | ✅               | ❌         | ❌        | ✅            | ❌            | ❌        |
| `.device`                   | ✅          | ✅            | ❌        | ❌     | ❌               | ❌         | ❌        | ❌            | ❌            | ❌        |
| `item()`                    | ✅          | ✅            | ✅        | ❌     | ✅               | ✅         | ❌        | ❌            | ✅            | ❌        |
| `__matmul__()`              | ✅          | ✅            | ❌        | ❌     | ✅               | ✅         | ❌        | ✅            | ❌            | ❌        |
| `__len__()`                 | ✅          | ✅            | ✅        | ✅     | ✅               | ✅         | ✅        | ✅            | ✅            | ✅        |
| `__iter__()`                | ✅          | ✅            | ✅        | ✅     | ✅               | ✅         | ✅        | ✅            | ✅            | ❌        |
| `iter()`                    | ROW        | ROW          | ROW      | ROW   | ROW             | ROW       | ROW      | COL NAME     | COL          | COL      |
| `__getitem__(index)`        | ROW        | ROW          | ROW      | ROW   | ROWS (⚡UNSAFE⚡) | ROW       | ROW      | ❌            | ROW          | COL      |
| `__getitem__(list[index])`  | ROWS       | ROWS         | ROWS     | ROWS  | ROWS (⚡UNSAFE⚡) | ROWS      | ❌        | COLS         | ROWS         | ❌        |
| `__getitem__(slice[index])` | ROWS       | ROWS         | ROWS     | ROWS  | ROWS            | ROWS      | ROWS     | ROWS         | ROWS         | ROWS     |
| `__getitem__(label)`        | ❌          | ❌            | ❌        | ❌     | ROW             | ❌         | ❌        | COL          | COL          | COL      |
| `__getitem__(list[label])`  | ❌          | ❌            | ❌        | ❌     | ROWS            | ❌         | ❌        | COLS         | COLS         | ❌        |
| `__getitem__(slice[label])` | ❌          | ❌            | ❌        | ❌     | ROWS            | ❌         | ❌        | ROWS         | COLS         | ❌        |
| `__getitem__(list[bool])`   | ROWS       | ROWS         | ROWS     | ROWS  | ROWS            | ❌         | ❌        | ROWS         | ❌            | ❌        |

Note:
    - The summary shows that `pyarrow` objects lack support for many operations.
    - mutability means that `x += 1` changes `x` in-place, resulting in object with identical id.
      Note that for instance `pl.Series` still supports `+=` but it does actually create a new object.

From this table, we are inclined to derive several protocols:

- `Table`-like protocol for 2d column-oriented arrays pd.DataFrame, pl.DataFrame and pa.Table.

Warning:
    `pandas.Series` has inherently usafe indexing! This is because `series[int]` and `series[list[int]]`

    - return the same as `series.iloc[int]` and `series.iloc[list[int]]` if series not indexed by integers.
    - return the same as `series.loc[int]` and `series.loc[list[int]]` if the series is indexed by integers.

    `polars` and `pyarrow` are strict by demanding that rows are indexed by indices, and columns by labels (strings).
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
    "SupportsMutation",
    "SupportsComparison",
    "SupportsMatmul",
    "ArrayKind",
    "SeriesKind",
    "TableKind",
    "NumericalArray",
    "NumericalSeries",
    "NumericalTensor",
    "MutableTensor",
]


from abc import abstractmethod
from collections.abc import Iterator
from typing import Any, Literal, Optional, Protocol, Self, overload, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from tsdm.types.aliases import Axis, MultiIndexer
from tsdm.types.scalars import BoolScalar

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
class SupportsMutation[Scalar](Protocol):
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

    Note:
        `pyarrow` does not support element-wise comparisons.

    References:
        - https://docs.python.org/3/c-api/buffer.html
        - https://numpy.org/doc/stable/reference/arrays.interface.html
        - https://numpy.org/devdocs/user/basics.interoperability.html
    """

    @property
    def shape(self) -> tuple[int, ...]: ...
    def __array__(self) -> NDArray: ...
    def __len__(self) -> int: ...


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

    NOTE: Many methods have subtle differences between backends:
     - `diff`: gives discrete differences for polars and pandas, but not for pyarrow
     - `value_counts`: polars returns a DataFrame, pandas a Series, pyarrow a StructArray
     - `unique`: polars and pyarrow return `Self`, pandas returns `np.ndarray`.

    References:
        - https://numpy.org/devdocs/user/basics.interoperability.html
    """

    def __array__(self) -> NDArray: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Scalar]: ...
    @overload
    def __getitem__(self, key: int, /) -> Scalar: ...
    @overload
    def __getitem__(self, key: slice, /) -> Self: ...

    def equals(self, other: Self) -> bool:
        r"""Check if the series is equal to another series."""
        ...


@runtime_checkable
class TableKind(Protocol):
    r"""A 2d column-oriented array with heterogenous data types.

    That it, it is a column-oriented 2d tensor which allows heterogenous data types.

    Note:
        In contrast to tensors (row-oriented), tables are column-oriented. Therefore,
        `__getitem__` returns a column, which is a SeriesKind, i.e. homogeneous 1d tensor.

    Note: The following methods differ between backends:
        - `pyarrow` does not support element-wise comparisons.
        - `iter`: yields columns for polars and pandas, but rows for pyarrow
             This is because `pyarrow` does not actually define `__iter__`.
        - `take`: not supported by polars
        - `columns`: pandas and polars return column names, pyarrow returns list of Arrays
        - `drop`: polars currently doing signature change
        - `filter`: pandas goes over columns, polars over rows

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

    @property
    def shape(self) -> tuple[int, int]: ...

    def __array__(self) -> NDArray[np.object_]: ...
    def __dataframe__(self, *, allow_copy: bool = True) -> object: ...
    def __len__(self) -> int: ...
    def __getitem__(self, key: str, /) -> SeriesKind: ...  # yields a column

    def equals(self, other: Self, /) -> bool:
        r"""Check if the table is equal to another table."""
        ...


@runtime_checkable
class NumericalArray[Scalar](ArrayKind[Scalar], Protocol):  # -Scalar
    r"""Subclass of `ArrayKind` that supports numerical operations.

    Examples:
        - `numpy.ndarray`
        - `pandas.DataFrame`
        - `pandas.Index`
        - `pandas.Series`
        - `pandas.extensions.ExtensionArray`
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

    # FIXME: change to __bool__ with torch==2.5.0
    def __contains__(self, element: Any, /) -> object: ...

    # NOTE: This is weakly typed since it returns different things on different objects.
    def __getitem__(self, key: Any, /) -> Self | Scalar: ...

    def all(self) -> Self | BoolScalar:
        r"""Return True if all elements are True."""
        ...

    def any(self) -> Self | BoolScalar:
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
    # NOTE: arrow/polars does not support complex data types!
    def __pow__(self, exponent: Self | float, /) -> Self: ...
    def __rpow__(self, base: Self | float, /) -> Self: ...
    # modulo %
    def __mod__(self, other: Self | Scalar | float, /) -> Self: ...
    def __rmod__(self, other: Self | Scalar | float, /) -> Self: ...

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


@runtime_checkable
class NumericalSeries[Scalar](NumericalArray[Scalar], Protocol):
    r"""Protocol for numerical series.

    Series are per definition one dimensional, and have a unique data type.
    Notably, this differs with respect to `NumericalTensor` by not supporting tuple-indexing.
    Moreover, its Iterator and `__getitem__(int)` return scalars.

    Note:
        Multidimensional Tensors are by definition Series of Tensors.
        For instance, a 3-dimensional numpy array is a `NumericalSeries[NDArray]`.

    Examples:
        - `numpy.ndarray` (if 1d)
        - `pandas.Index`
        - `pandas.Series`
        - `pandas.extensions.ExtensionArray`
        - `polars.Series`
        - `torch.Tensor` (if 1d)

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


@runtime_checkable
class NumericalTensor[Scalar](NumericalArray[Scalar], Protocol):
    r"""Protocol for numerical tensors.

    Compared to `NumericalSeries`, tensors *can* have multiple dimensions, and
    must support more Indexing operations, In particular `...` (Ellipsis), and
    tuples of ints and/or slices.

    Examples:
        - `numpy.ndarray`
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

    def __iter__(self) -> Iterator[Scalar] | Iterator[Self]: ...

    # matrix multiplication @
    def __matmul__(self, other: Self, /) -> Self: ...
    def __rmatmul__(self, other: Self, /) -> Self: ...

    # fmt: off
    @overload  # depending on Tensor Rank, can return Scalar or Tensor
    def __getitem__(self, key: int | tuple[int, ...], /) -> Scalar | Self: ...  # type: ignore[overload-overlap]
    @overload
    def __getitem__(self, key: Self | MultiIndexer, /) -> Self: ...
    # fmt: on

    def argmin(self, axis: Optional[int] = None, /) -> int | Self:
        r"""Return the index of the minimum value."""
        ...

    def argmax(self, axis: Optional[int] = None, /) -> int | Self:
        r"""Return the index of the maximum value."""
        ...

    def argsort(self, axis: int = -1, /) -> Self:
        r"""Return the indices that would sort the tensor (along last axis)."""
        ...

    def clip(self, lower: Any, upper: Any, /) -> Self: ...
    def cumsum(self, axis: int, /) -> Scalar | Self: ...
    def cumprod(self, axis: int, /) -> Scalar | Self: ...

    def ravel(self) -> Self:
        r"""Return a flattened version of the tensor."""
        ...

    def take(self, indices: Any, /) -> Self:
        r"""Select elements from the array by index."""
        ...

    def item(self) -> Scalar:
        r"""Return the scalar value the tensor if it only has a single element.

        Otherwise, raises `ValueError`.
        """
        ...

    def std(self, axis: Axis = ..., /) -> Scalar | Self: ...
    def var(self, axis: Axis = ..., /) -> Scalar | Self: ...

    # region stupid overloads ----------------------------------------------------------
    # FIXME: https://github.com/python/typing/discussions/1782
    @overload
    def mean(self) -> Scalar | Self: ...
    @overload
    def mean(self, axis: Axis, /) -> Scalar | Self: ...

    @overload
    def sum(self) -> Self: ...
    @overload
    def sum(self, axis: Axis, /) -> Self: ...

    @overload
    def prod(self) -> Scalar | Self: ...
    @overload
    def prod(self, axis: int, /) -> Scalar | Self: ...

    @overload
    def round(self) -> Self: ...
    @overload
    def round(self, *, decimals: int) -> Self: ...

    @overload
    def squeeze(self) -> Self: ...
    @overload  # FIXME: https://github.com/pytorch/pytorch/issues/137422
    def squeeze(self, axis: int, /) -> Self: ...

    # endregion stupid overloads -------------------------------------------------------


@runtime_checkable
class MutableTensor[Scalar](NumericalTensor[Scalar], SupportsMutation, Protocol):
    r"""Subclass of `NumericalTensor` that supports inplace operations.

    Examples:
        - `numpy.ndarray`
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
