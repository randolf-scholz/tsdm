r"""Small mixin protocols."""

__all__ = [
    "SupportsArray",
    "SupportsArrayUfunc",
    "SupportsDataFrame",
    "SupportsDevice",
    "SupportsDtype",
    "SupportsGetItem",
    "SupportsItem",
    "SupportsKeysAndGetItem",
    "SupportsLenAndGetItem",
    "SupportsNdim",
    "SupportsRound",
    "SupportsShape",
    "SupportsSlicing",
]


from abc import abstractmethod
from collections.abc import Collection
from typing import Any, Literal, Protocol, Self, overload, runtime_checkable

import numpy as np
from numpy.typing import NDArray


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
    @abstractmethod
    def round(self) -> Self: ...
    @overload
    @abstractmethod
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


@runtime_checkable
class SupportsSlicing[V](SupportsGetItem[int, V], Protocol):  # +V
    r"""Protocol for objects that support slicing with integer indices."""

    @overload
    @abstractmethod
    def __getitem__(self, index: int, /) -> V: ...
    @overload
    @abstractmethod
    def __getitem__(self, index: slice, /) -> Self: ...
