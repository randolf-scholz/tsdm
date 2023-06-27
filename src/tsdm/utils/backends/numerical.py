"""Numerical functions with shared signatures for multiple backends."""

from __future__ import annotations

__all__ = ["Backend", "get_backend", "KernelProvider", "is_singleton", "is_scalar"]

from math import prod
from typing import Any, Callable, Generic, Literal, Protocol, TypeAlias, TypeVar, cast

import numpy
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from numpy import ndarray
from pandas import DataFrame, Series
from torch import Tensor
from typing_extensions import get_args

from tsdm.types.aliases import Axes
from tsdm.types.protocols import SupportsShape
from tsdm.utils.backends.pandas import (
    pandas_clip,
    pandas_nanmax,
    pandas_nanmean,
    pandas_nanmin,
    pandas_nanstd,
    pandas_where,
)
from tsdm.utils.backends.torch import torch_nanmax, torch_nanmin, torch_nanstd

T = TypeVar("T", Series, DataFrame, Tensor, ndarray)
Backend: TypeAlias = Literal["torch", "numpy", "pandas"]


class SupportsAxis(Protocol[T]):
    """A protocol for callables that support the `axes` keyword argument."""

    def __call__(self, x: T, /, axis: Axes = None) -> T:
        ...


class SupportsKeepdims(Protocol[T]):
    """A protocol for callables that support the `axes` and `keepdims` keyword arguments."""

    def __call__(self, x: T, /, axis: Axes = None, keepdims: bool = False) -> T:
        ...


def get_backend(obj: object, fallback: Backend = "numpy") -> Backend:
    """Get the backend of a set of objects."""
    types: set[Backend] = set()

    match obj:
        case tuple() | set() | frozenset() | list() as container:
            types |= {get_backend(o) for o in container}
        case dict() as mapping:
            types |= {get_backend(o) for o in mapping.values()}
        case Tensor():
            types.add("torch")
        case DataFrame() | Series():  # type: ignore[misc]
            types.add("pandas")  # type: ignore[unreachable]
        case ndarray():
            types.add("numpy")
        case _:
            pass

    match len(types):
        case 0:
            return fallback
        case 1:
            return types.pop()
        case _:
            raise ValueError(f"More than 1 backend detected: {types}.")


class KernelProvider(Generic[T]):
    """Provides kernels for numerical operations."""

    backend: Backend

    def __init__(self, backend: str) -> None:
        assert backend in get_args(Backend)
        self.backend = cast(Backend, backend)

    @property
    def where(self) -> Callable[[T, T, T], T]:
        kernels = {
            "numpy": numpy.where,
            "pandas": pandas_where,
            "torch": torch.where,
        }
        return kernels[self.backend]  # type: ignore[return-value]

    @property
    def clip(self) -> Callable[[T, T | None, T | None], T]:
        kernels = {
            "numpy": numpy.clip,
            "pandas": pandas_clip,
            "torch": torch.clip,
        }
        return kernels[self.backend]  # type: ignore[return-value]

    @property
    def to_tensor(self) -> Callable[[Any], T]:
        kernels = {
            "numpy": numpy.array,
            "pandas": numpy.array,
            "torch": torch.tensor,
        }
        return kernels[self.backend]  # type: ignore[return-value]

    @property
    def nanmin(self) -> SupportsAxis[T]:
        kernels = {
            "numpy": np.nanmin,
            "pandas": pandas_nanmin,
            "torch": torch_nanmin,
        }
        return kernels[self.backend]  # type: ignore[return-value]

    @property
    def nanmax(self) -> SupportsAxis[T]:
        kernels = {
            "numpy": np.nanmax,
            "pandas": pandas_nanmax,
            "torch": torch_nanmax,
        }
        return kernels[self.backend]  # type: ignore[return-value]

    @property
    def nanmean(self) -> SupportsAxis[T]:
        kernels = {
            "numpy": np.nanmean,
            "pandas": pandas_nanmean,
            "torch": torch.nanmean,
        }
        return kernels[self.backend]  # type: ignore[return-value]

    @property
    def nanstd(self) -> SupportsAxis[T]:
        kernels = {
            "numpy": np.nanstd,
            "pandas": pandas_nanstd,
            "torch": torch_nanstd,
        }
        return kernels[self.backend]  # type: ignore[return-value]


def is_singleton(x: SupportsShape) -> bool:
    """Determines whether a tensor like object has a single element."""
    return prod(x.shape) == 1
    # numpy: size, len  / shape + prod
    # torch: size + prod / numel / shape + prod
    # table: shape + prod
    # DataFrame: shape + prod
    # Series: shape + prod
    # pyarrow table: shape + prod
    # pyarrow array: ????


def is_scalar(x: Any) -> bool:
    """Determines whether an object is a scalar."""
    return (
        isinstance(x, (int, float, str, bool))
        or np.isscalar(x)
        or pd.api.types.is_scalar(x)
        or isinstance(x, pa.Scalar)
    )


def to_scalar():
    """Convert a singleton to a scalar."""
    raise NotImplementedError
