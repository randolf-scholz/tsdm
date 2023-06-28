"""Numerical functions with shared signatures for multiple backends."""

from __future__ import annotations

__all__ = [
    # Classes
    "BackendID",
    "Kernels",
    "Backend",
    # Functions
    "get_backend",
    "is_scalar",
    "to_scalar",
]

from collections.abc import Mapping
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    ParamSpec,
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

from tsdm.backend.pandas import (
    pandas_clip,
    pandas_nanmax,
    pandas_nanmean,
    pandas_nanmin,
    pandas_nanstd,
    pandas_where,
)
from tsdm.backend.torch import torch_nanmax, torch_nanmin, torch_nanstd
from tsdm.backend.universal import (
    false_like as universal_false_like,
    true_like as universal_true_like,
)
from tsdm.types.aliases import Axes
from tsdm.types.protocols import SelfMap, SelfMapProto

P = ParamSpec("P")

T = TypeVar("T")
"""A type variable for numerical objects."""
BackendID: TypeAlias = Literal["torch", "numpy", "pandas"]
"""A type alias for the supported backends."""


class ContractionProto(Protocol[T]):
    """A protocol for callables that support the `axes` keyword argument."""

    def __call__(self, __x: T, *, axis: Axes = None) -> T:
        ...


class ClipProto(Protocol[T]):
    """Protocol for Clip functions."""

    def __call__(self, __x: T, __lower: T | None, __upper: T | None) -> T:
        ...


class WhereProto(Protocol[T]):
    """Protocol for Where functions."""

    def __call__(self, __cond: T, __x: T, __y: T) -> T:
        ...


class Contraction(Protocol):
    """A protocol for callables that support the `axes` keyword argument."""

    def __call__(self, __x: T, *, axis: Axes = None) -> T:
        ...


class Clip(Protocol):
    """Protocol for Clip functions."""

    def __call__(self, __x: T, __lower: T | None, __upper: T | None) -> T:
        ...


class Where(Protocol):
    """Protocol for Where functions."""

    def __call__(self, __cond: T, __x: T, __y: T) -> T:
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
    """A class that provides backend kernels."""

    clip: Mapping[BackendID, ClipProto] = {
        "numpy": numpy.clip,
        "pandas": pandas_clip,
        "torch": torch.clip,
    }

    nanmin: Mapping[BackendID, ContractionProto] = {
        "numpy": np.nanmin,
        "pandas": pandas_nanmin,
        "torch": torch_nanmin,
    }

    nanmax: Mapping[BackendID, ContractionProto] = {
        "numpy": np.nanmax,
        "pandas": pandas_nanmax,
        "torch": torch_nanmax,
    }

    nanmean: Mapping[BackendID, ContractionProto] = {
        "numpy": np.nanmean,
        "pandas": pandas_nanmean,
        "torch": torch.nanmean,  # type: ignore[dict-item]
    }

    nanstd: Mapping[BackendID, ContractionProto] = {
        "numpy": np.nanstd,
        "pandas": pandas_nanstd,
        "torch": torch_nanstd,
    }

    false_like: Mapping[BackendID, SelfMapProto] = {
        "numpy": universal_false_like,
        "pandas": universal_false_like,
        "torch": universal_false_like,
    }

    true_like: Mapping[BackendID, SelfMapProto] = {
        "numpy": universal_true_like,
        "pandas": universal_true_like,
        "torch": universal_true_like,
    }

    to_tensor: Mapping[BackendID, SelfMapProto] = {
        "numpy": numpy.array,
        "pandas": numpy.array,
        "torch": torch.tensor,
    }

    where: Mapping[BackendID, WhereProto] = {
        "numpy": numpy.where,
        "pandas": pandas_where,
        "torch": torch.where,
    }


class Backend(Generic[T]):
    """Provides kernels for numerical operations."""

    selected_backend: BackendID

    # KERNELS
    to_tensor: Callable[[Any], T]

    clip: Clip
    where: Where

    nanmax: Contraction
    nanmean: Contraction
    nanmin: Contraction
    nanstd: Contraction

    true_like: SelfMap
    false_like: SelfMap

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
