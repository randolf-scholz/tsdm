"""Numerical functions with shared signatures for multiple backends."""

from __future__ import annotations

__all__ = [
    # Classes
    "BackendID",
    "Kernels",
    "KernelProvider",
    # Functions
    "get_backend",
    "is_scalar",
    "to_scalar",
]

from typing import (
    Any,
    Callable,
    Final,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
)

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
from tsdm.utils.backends.pandas import (
    pandas_clip,
    pandas_nanmax,
    pandas_nanmean,
    pandas_nanmin,
    pandas_nanstd,
    pandas_where,
)
from tsdm.utils.backends.torch import torch_nanmax, torch_nanmin, torch_nanstd
from tsdm.utils.backends.universal import false_like as universal_false_like
from tsdm.utils.backends.universal import true_like as universal_true_like

T = TypeVar("T", Series, DataFrame, Tensor, ndarray)
"""A type variable for numerical objects."""
BackendID: TypeAlias = Literal["torch", "numpy", "pandas"]
"""A type alias for the supported backends."""


class SupportsAxis(Protocol[T]):
    """A protocol for callables that support the `axes` keyword argument."""

    def __call__(self, x: T, /, axis: Axes = None) -> T:
        ...


class SupportsKeepdims(Protocol[T]):
    """A protocol for callables that support the `axes` and `keepdims` keyword arguments."""

    def __call__(self, x: T, /, axis: Axes = None, keepdims: bool = False) -> T:
        ...


def get_backend(obj: object, fallback: BackendID = "numpy") -> BackendID:
    """Get the backend of a set of objects."""
    types: set[BackendID] = set()

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


class Kernels:
    """A class that holds the kernels for each backend."""

    where: Final[dict[BackendID, Callable[[T, T, T], T]]] = {
        "numpy": numpy.where,
        "pandas": pandas_where,
        "torch": torch.where,
    }

    clip: Final[dict[BackendID, Callable[[T, T | None, T | None], T]]] = {
        "numpy": numpy.clip,
        "pandas": pandas_clip,
        "torch": torch.clip,
    }

    to_tensor: Final[dict[BackendID, Callable[[Any], T]]] = {
        "numpy": numpy.array,
        "pandas": numpy.array,
        "torch": torch.tensor,
    }

    nanmin: Final[dict[BackendID, SupportsAxis[T]]] = {
        "numpy": np.nanmin,
        "pandas": pandas_nanmin,
        "torch": torch_nanmin,
    }

    nanmax: Final[dict[BackendID, SupportsAxis[T]]] = {
        "numpy": np.nanmax,
        "pandas": pandas_nanmax,
        "torch": torch_nanmax,
    }

    nanmean: Final[dict[BackendID, SupportsAxis[T]]] = {
        "numpy": np.nanmean,
        "pandas": pandas_nanmean,
        "torch": torch.nanmean,
    }

    nanstd: Final[dict[BackendID, SupportsAxis[T]]] = {
        "numpy": np.nanstd,
        "pandas": pandas_nanstd,
        "torch": torch_nanstd,
    }

    true_like: Final[dict[BackendID, Callable[[T], T]]] = {
        "numpy": universal_true_like,
        "pandas": universal_true_like,
        "torch": universal_true_like,
    }

    false_like: Final[dict[BackendID, Callable[[T], T]]] = {
        "numpy": universal_false_like,
        "pandas": universal_false_like,
        "torch": universal_false_like,
    }


class KernelProvider(Generic[T]):
    """Provides kernels for numerical operations."""

    selected_backend: BackendID

    # KERNELS
    clip: Callable[[T, T | None, T | None], T]
    false_like: Callable[[T], T]
    nanmax: SupportsAxis[T]
    nanmean: SupportsAxis[T]
    nanmin: SupportsAxis[T]
    nanstd: SupportsAxis[T]
    to_tensor: Callable[[Any], T]
    true_like: Callable[[T], T]
    where: Callable[[T, T, T], T]

    def __init__(self, backend: str) -> None:
        assert backend in get_args(BackendID)
        self.selected_backend = cast(BackendID, backend)

        self.clip = Kernels.clip[self.selected_backend]
        self.false_like = Kernels.false_like[self.selected_backend]
        self.nanmax = Kernels.nanmax[self.selected_backend]
        self.nanmean = Kernels.nanmean[self.selected_backend]
        self.nanmin = Kernels.nanmin[self.selected_backend]
        self.nanstd = Kernels.nanstd[self.selected_backend]
        self.to_tensor = Kernels.to_tensor[self.selected_backend]
        self.true_like = Kernels.true_like[self.selected_backend]
        self.where = Kernels.where[self.selected_backend]

    def switch_backend(self, backend: str) -> None:
        self.__init__(backend)


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
