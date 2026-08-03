r"""Alias types for numerical computations."""

__all__ = [
    "PythonScalar",
    "Axis",
    "Dims",
    "Size",
    "Shape",
    # Comparison operations
    "SupportsEquality",
    "SupportsComparison",
    # Mixins
    "SupportsArray",
    "SupportsArrayUfunc",
    "SupportsDataFrame",
    "SupportsDevice",
    "SupportsDtype",
    "SupportsItem",
    "SupportsNdim",
    "SupportsRound",
    "SupportsShape",
]

from abc import abstractmethod
from datetime import datetime, timedelta
from typing import (
    Any,
    Literal,
    Protocol,
    Self,
    overload,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

type PythonScalar = bool | int | float | complex | str | bytes | datetime | timedelta
r"""Type Alias for Python scalars."""
type Axis = None | int | tuple[int, ...]
r"""Type Alias for axestype ."""
type Dims = None | int | list[int]
r"""Type Alias for dimensions compatible with torchscript."""  # FIXME: https://github.com/pytorch/pytorch/issues/64700
type Size = int | tuple[int, ...]
r"""Type Alias for size-like objects (note: `sample(size=None)` creates scalar."""
type Shape = int | tuple[int, ...]
r"""Type Alias for shape-like objects (note: `ones(shape=None)` creates 0d-array."""


@runtime_checkable
class SupportsShape(Protocol):
    r"""Protocol for objects that support shape."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]: ...


@runtime_checkable
class SupportsNdim(Protocol):
    r"""We just test for ndim, since e.g. tf.Tensor does not have ndim."""

    @property
    @abstractmethod
    def ndim(self) -> int: ...


@runtime_checkable
class SupportsDevice[DeviceT = Any](Protocol):
    r"""Protocol for objects that support `device`."""

    @property
    @abstractmethod
    def device(self) -> DeviceT: ...


@runtime_checkable
class SupportsDtype[DTypeT = Any](Protocol):
    r"""We just test for dtype, since e.g. tf.Tensor does not have ndim.

    Examples:
        - `numpy.ndarray`
        - `pandas.Series`
        - `polars.Series`
        - `torch.Tensor`

    Examples: (Counter-Examples)
        - `pandas.DataFrame`
        - `polars.DataFrame`
        - `pyarrow.Array`
    """

    @property
    @abstractmethod
    def dtype(self) -> DTypeT: ...


@runtime_checkable
class SupportsArray[ScalarT: np.generic](Protocol):
    r"""Protocol for objects that support `__array__`.

    References:
        https://numpy.org/doc/stable/reference/c-api/array.html
    """

    @abstractmethod
    def __array__(self) -> NDArray[ScalarT]:
        r"""Return the array of the tensor."""
        ...


@runtime_checkable
class SupportsArrayUfunc(SupportsArray, Protocol):
    r"""Protocol for objects that support `__array_ufunc__`.

    Notably, numpy functions like `numpy.exp` can be directly applied to such objects.
    The main example are `pandas.Series` and `pandas.DataFrame`.

    Examples:
        - `numpy.ndarray`
        - `pandas.DataFrame`
        - `pandas.Index`
        - `pandas.Series`
        - `polars.Series`

    Examples: (Couter-Examples)
        - `polars.DataFrame`
        - `pyarrow.Array`
        - `pyarrow.Table`
        - `torch.Tensor`

    References:
        https://numpy.org/doc/stable/reference/ufuncs.html
    """

    @abstractmethod
    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "at"],
        /,
        *inputs: Any,
        **kwargs: Any,
    ) -> Self:
        r"""Return the array resulting from applying the ufunc."""
        ...


@runtime_checkable
class SupportsDataFrame[DataFrameT = Any](Protocol):
    r"""Protocol for objects that support `__dataframe__`.

    References:
        https://data-apis.org/dataframe-protocol/latest/index.html
    """

    @abstractmethod
    def __dataframe__(self) -> DataFrameT: ...


@runtime_checkable
class SupportsItem[Scalar](Protocol):  # +T
    r"""Protocol for objects that support `.item()`.

    Return the scalar value the tensor if it only has a single element.

    If the tensor has more than one element, raise an error.
    """

    @abstractmethod
    def item(self) -> Scalar: ...


@runtime_checkable
class SupportsRound(Protocol):
    r"""Protocol for objects that support `round`."""

    # FIXME: https://github.com/python/typing/discussions/1782
    @overload
    @abstractmethod
    def round(self) -> Self: ...
    @overload
    @abstractmethod
    def round(self, *, decimals: int) -> Self: ...


class SupportsEquality[ResultT](Protocol):
    r"""Protocol for objects that support equality and inequality comparisons."""

    # Note: really no good choice for the argument type here,
    #   as most built-ins will require object argument, but DSLs will restrict to their own type.
    #   so just use Any.
    # equality ==
    def __eq__(self, other: Any, /) -> ResultT: ...  # type: ignore[override]
    # inequality !=
    def __ne__(self, other: Any, /) -> ResultT: ...  # type: ignore[override]


class SupportsComparison[ComparableT, ResultT](Protocol):
    r"""Protocol for objects that support comparison operations."""

    # comparisons (element-wise)
    def __le__(self, other: Self | ComparableT, /) -> ResultT: ...
    def __ge__(self, other: Self | ComparableT, /) -> ResultT: ...
    def __lt__(self, other: Self | ComparableT, /) -> ResultT: ...
    def __gt__(self, other: Self | ComparableT, /) -> ResultT: ...
