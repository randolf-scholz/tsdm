r"""Outdated: Protocols for numerical array-like types."""

__all__ = [
    "SupportsMutation",
    "NumericalArray",
    "NumericalSeries",
    "NumericalTensor",
    "MutableTensor",
]


from collections.abc import Iterator
from types import EllipsisType
from typing import (
    Any,
    Protocol,
    Self,
    SupportsInt,
    overload,
    runtime_checkable,
)

from tsdm.experimental.types.arrays import BaseArray
from tsdm.experimental.types.mixins import SupportsComparison
from tsdm.experimental.types.scalars import BoolScalar

type Axis = None | int | tuple[int, ...]
r"""Type Alias for axestype ."""
type Size = int | tuple[int, ...]
r"""Type Alias for size-like objects (note: `sample(size=None)` creates scalar."""
type DimArg = None | int | list[int]
r"""Type Alias for dimensions compatible with torchscript."""
type IndexArg1D = None | int | slice | range | list[int] | list[bool] | EllipsisType
r"""Type alias for `__getitem__` argument for tensors."""
type IndexArgND = IndexArg1D | tuple[IndexArg1D, ...]
r"""Indexer that always returns a sub-tensor."""


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


@runtime_checkable
class NumericalArray[Scalar](
    BaseArray,
    SupportsComparison,
    Protocol,
):
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

    def all(self) -> BoolScalar: ...
    def any(self) -> BoolScalar: ...
    def min(self) -> Any: ...
    def max(self) -> Any: ...

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

    # FIXME: Possibly should look like
    #   def __add__[Z](self, other: Y | Vec[Y], /) -> Vec[Z]: ...
    #   where Y = SupportsRAdd[Scalar, Z] is a protocol with
    #   def __radd__(self, other: Scalar, /) -> Z: ...
    #   this would allow compatibility for timedelta/datetime.

    # region binary operations ---------------------------------------------------------
    # FIXME: https://github.com/python/typing/issues/2021
    #   Because of the current spec, we need overloads instead of simply unions.
    # + (addition)
    def __add__(self, other: Self | Scalar, /) -> Self: ...
    def __radd__(self, other: Self | Scalar, /) -> Self: ...
    # - (subtraction)
    def __sub__(self, other: Self | Scalar, /) -> Self: ...
    def __rsub__(self, other: Self | Scalar, /) -> Self: ...
    # * (multiplication)
    def __mul__(self, other: Self | Scalar, /) -> Self: ...
    def __rmul__(self, other: Self | Scalar, /) -> Self: ...
    # / (true division)
    def __truediv__(self, other: Self | Scalar, /) -> Self: ...
    def __rtruediv__(self, other: Self | Scalar, /) -> Self: ...
    # // (floor division)
    def __floordiv__(self, other: Self | Scalar, /) -> Self: ...
    def __rfloordiv__(self, other: Self | Scalar, /) -> Self: ...
    # ** (power)
    # NOTE: arrow/polars does not support complex data types!
    def __pow__(self, exponent: Self | float, /) -> Self: ...
    def __rpow__(self, base: Self | float, /) -> Self: ...
    # % (modulo)
    def __mod__(self, other: Self | Scalar | float, /) -> Self: ...
    def __rmod__(self, other: Self | Scalar | float, /) -> Self: ...
    # & (AND)
    def __and__(self, other: Self | Scalar, /) -> Self: ...
    def __rand__(self, other: Self | Scalar, /) -> Self: ...
    # | (OR)
    def __or__(self, other: Self | Scalar, /) -> Self: ...
    def __ror__(self, other: Self | Scalar, /) -> Self: ...
    # ^ (XOR)
    def __xor__(self, other: Self | Scalar, /) -> Self: ...
    def __rxor__(self, other: Self | Scalar, /) -> Self: ...

    # endregion binary operations ------------------------------------------------------

    # region overload alternatives -----------------------------------------------------
    # >>> # + (addition)
    # >>> @overload
    # >>> def __add__(self, other: Self, /) -> Self: ...
    # >>> @overload
    # >>> def __add__(self, other: Scalar, /) -> Self: ...
    # >>> @overload
    # >>> def __radd__(self, other: Self, /) -> Self: ...
    # >>> @overload
    # >>> def __radd__(self, other: Scalar, /) -> Self: ...
    # >>> # - (subtraction)
    # >>> @overload
    # >>> def __sub__(self, other: Self, /) -> Self: ...
    # >>> @overload
    # >>> def __sub__(self, other: Scalar, /) -> Self: ...
    # >>> @overload
    # >>> def __rsub__(self, other: Self, /) -> Self: ...
    # >>> @overload
    # >>> def __rsub__(self, other: Scalar, /) -> Self: ...
    # >>> # * (multiplication)
    # >>> @overload
    # >>> def __mul__(self, other: Self, /) -> Self: ...
    # >>> @overload
    # >>> def __mul__(self, other: Scalar, /) -> Self: ...
    # >>> @overload
    # >>> def __mul__(self, other: float, /) -> Self: ...
    # >>> @overload
    # >>> def __rmul__(self, other: Self, /) -> Self: ...
    # >>> @overload
    # >>> def __rmul__(self, other: Scalar, /) -> Self: ...
    # >>> @overload
    # >>> def __rmul__(self, other: float, /) -> Self: ...
    # >>> # / (true division)
    # >>> @overload
    # >>> def __truediv__(self, other: Self, /) -> Self: ...
    # >>> @overload
    # >>> def __truediv__(self, other: Scalar, /) -> Self: ...
    # >>> @overload
    # >>> def __truediv__(self, other: int, /) -> Self: ...
    # >>> @overload
    # >>> def __rtruediv__(self, other: Self, /) -> Self: ...
    # >>> @overload
    # >>> def __rtruediv__(self, other: Scalar, /) -> Self: ...
    # >>> @overload
    # >>> def __rtruediv__(self, other: int, /) -> Self: ...
    # >>> # // (floor division)
    # >>> # NOTE: complex types do not support floor division.
    # >>> @overload
    # >>> def __floordiv__(self, other: Self, /) -> Self: ...
    # >>> @overload
    # >>> def __floordiv__(self, other: Scalar, /) -> Self: ...
    # >>> @overload
    # >>> def __rfloordiv__(self, other: Self, /) -> Self: ...
    # >>> @overload
    # >>> def __rfloordiv__(self, other: Scalar, /) -> Self: ...
    # >>> # ** (power)
    # >>> # NOTE: arrow/polars does not support complex data types!
    # >>> @overload
    # >>> def __pow__(self, exponent: Self, /) -> Self: ...
    # >>> @overload
    # >>> def __pow__(self, exponent: float, /) -> Self: ...
    # >>> @overload
    # >>> def __rpow__(self, base: Self, /) -> Self: ...
    # >>> @overload
    # >>> def __rpow__(self, base: float, /) -> Self: ...
    # >>> # % (modulo)
    # >>> @overload
    # >>> def __mod__(self, other: Self, /) -> Self: ...
    # >>> @overload
    # >>> def __mod__(self, other: Scalar, /) -> Self: ...
    # >>> @overload
    # >>> def __mod__(self, other: float, /) -> Self: ...
    # >>> @overload
    # >>> def __rmod__(self, other: Self, /) -> Self: ...
    # >>> @overload
    # >>> def __rmod__(self, other: Scalar, /) -> Self: ...
    # >>> @overload
    # >>> def __rmod__(self, other: float, /) -> Self: ...
    # >>> # & (AND)
    # >>> @overload
    # >>> def __and__(self, other: Self, /) -> Self: ...
    # >>> @overload
    # >>> def __and__(self, other: Scalar, /) -> Self: ...
    # >>> @overload
    # >>> def __and__(self, other: int, /) -> Self: ...
    # >>> @overload
    # >>> def __rand__(self, other: Self, /) -> Self: ...
    # >>> @overload
    # >>> def __rand__(self, other: Scalar, /) -> Self: ...
    # >>> @overload
    # >>> def __rand__(self, other: int, /) -> Self: ...
    # >>> # | (OR)
    # >>> @overload
    # >>> def __or__(self, other: Self, /) -> Self: ...
    # >>> @overload
    # >>> def __or__(self, other: Scalar, /) -> Self: ...
    # >>> @overload
    # >>> def __or__(self, other: int, /) -> Self: ...
    # >>> @overload
    # >>> def __ror__(self, other: Self, /) -> Self: ...
    # >>> @overload
    # >>> def __ror__(self, other: Scalar, /) -> Self: ...
    # >>> @overload
    # >>> def __ror__(self, other: int, /) -> Self: ...
    # >>> # ^ (XOR)
    # >>> @overload
    # >>> def __xor__(self, other: Self, /) -> Self: ...
    # >>> @overload
    # >>> def __xor__(self, other: Scalar, /) -> Self: ...
    # >>> @overload
    # >>> def __xor__(self, other: int, /) -> Self: ...
    # >>> @overload
    # >>> def __rxor__(self, other: Self, /) -> Self: ...
    # >>> @overload
    # >>> def __rxor__(self, other: Scalar, /) -> Self: ...
    # >>> @overload
    # >>> def __rxor__(self, other: int, /) -> Self: ...
    # endregion overload alternatives --------------------------------------------------

    # endregion arithmetic operations --------------------------------------------------


@runtime_checkable
class NumericalSeries[Scalar](
    NumericalArray[Scalar],
    Protocol,
):
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
    def __iter__(self) -> Iterator[Scalar] | Iterator[Self]: ...

    # fmt: off
    @overload
    def __getitem__(self, key: int, /) -> Scalar | Self: ...
    @overload
    def __getitem__(self, key: slice | range | list[int] | Self, /) -> Self: ...
    # fmt: on


@runtime_checkable
class NumericalTensor[Scalar](
    NumericalArray[Scalar],
    Protocol,
):
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
    def __getitem__(self, key: int | tuple[int, ...], /) -> Scalar | Self: ...  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
    @overload
    def __getitem__(self, key: Self | IndexArgND, /) -> Self: ...
    # fmt: on

    @overload
    def argmin(self, axis: None = ..., /) -> SupportsInt: ...
    @overload
    def argmin(self, axis: int, /) -> Self: ...

    @overload
    def argmax(self, axis: None = ..., /) -> SupportsInt: ...
    @overload
    def argmax(self, axis: int, /) -> Self: ...

    def argsort(self, axis: int = -1, /) -> Self: ...

    def clip(self, lower: Any, upper: Any, /) -> Self: ...
    def cumsum(self, axis: int, /) -> Scalar | Self: ...
    def cumprod(self, axis: int, /) -> Scalar | Self: ...
    def std(self, axis: Axis = ..., /) -> Scalar | Self: ...
    def var(self, axis: Axis = ..., /) -> Scalar | Self: ...

    def ravel(self) -> Self:
        r"""Return a flattened version of the tensor."""
        ...

    def take(self, indices: Any, /) -> Self:
        r"""Select elements from the array by index."""
        ...

    # NOTE: Disabled since pyright says numpy incompatible
    # def item(self) -> Scalar:
    #     r"""Return the scalar value the tensor if it only has a single element.
    #
    #     Otherwise, raises `ValueError`.
    #     """
    #     ...

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
class MutableTensor[Scalar](
    NumericalTensor[Scalar],
    SupportsMutation,
    Protocol,
):
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
