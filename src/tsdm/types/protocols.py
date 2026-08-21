r"""Small mixin protocols."""

__all__ = [
    # Protocols
    "SupportsBool",
    "SupportsGetItem",
    "SupportsKeysAndGetItem",
    "SupportsLenAndGetItem",
    "SupportsSlicing",
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
    # functions
    "implements",
]


from abc import abstractmethod
from collections.abc import Collection
from types import GenericAlias
from typing import (
    Any,
    Literal,
    Protocol,
    Self,
    get_origin,
    is_protocol,
    overload,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray


class SupportsEquality[ResultT](Protocol):
    r"""Protocol for objects that support equality and inequality comparisons."""

    # Note: really no good choice for the argument type here,
    #   as most built-ins will require object argument, but DSLs will restrict to their own type.
    #   so just use Any.
    # equality ==
    def __eq__(self, other: Any, /) -> ResultT: ...  # type: ignore
    # inequality !=
    def __ne__(self, other: Any, /) -> ResultT: ...  # type: ignore


class SupportsComparison[ComparableT, ResultT](Protocol):
    r"""Protocol for objects that support comparison operations."""

    # comparisons (element-wise)
    def __le__(self, other: Self | ComparableT, /) -> ResultT: ...
    def __ge__(self, other: Self | ComparableT, /) -> ResultT: ...
    def __lt__(self, other: Self | ComparableT, /) -> ResultT: ...
    def __gt__(self, other: Self | ComparableT, /) -> ResultT: ...


@runtime_checkable
class SupportsBool(Protocol):
    r"""Protocol for types that support boolean operations."""

    def __bool__(self) -> bool: ...


@runtime_checkable
class SupportsGetItem[K, V](Protocol):  # -K, +V
    r"""Protocol for objects that support `__getitem__`."""

    @abstractmethod
    def __getitem__(self, key: K, /) -> V: ...


@runtime_checkable
class SupportsKeysAndGetItem[K, V](Protocol):  # K, +V
    r"""Protocol for objects that support `__getitem__` and `keys`."""

    @abstractmethod
    def keys(self) -> Collection[K]: ...
    @abstractmethod
    def __getitem__(self, key: K, /) -> V: ...


@runtime_checkable
class SupportsLenAndGetItem[V](Protocol):  # +V
    r"""Protocol for objects that support integer based `__getitem__` and `__len__`."""

    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __getitem__(self, index: int, /) -> V: ...


class SupportsSlicing[V](SupportsGetItem[int, V], Protocol):  # +V
    r"""Protocol for objects that support slicing with integer indices."""

    @overload
    @abstractmethod
    def __getitem__(self, index: int, /) -> V: ...
    @overload
    @abstractmethod
    def __getitem__(self, index: slice, /) -> Self: ...


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


def implements[T](cls: type[T], /, *protos: type | GenericAlias) -> type[T]:
    r"""Check if a class implements a protocol."""
    for proto in protos:
        if isinstance(proto, GenericAlias):
            proto = get_origin(proto)  # ruff: ignore[PLW2901]

        if not (isinstance(proto, type) and is_protocol(proto)):
            raise TypeError(f"{proto=} is not a protocol type.")

        if bool(getattr(proto, "_is_runtime_protocol", False)):
            assert issubclass(cls, proto)

    return cls
