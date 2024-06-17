r"""Numerical functions with shared signatures for multiple backends."""

__all__ = [
    # CONSTANTS
    "BACKENDS",
    # Classes
    "BackendID",
    "Kernels",
    "Backend",
    # Functions
    "get_backend",
    "get_backend_id",
    "gather_types",
]

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import EllipsisType, NotImplementedType

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import torch
from numpy import ndarray
from torch import Tensor
from typing_extensions import (
    Final,
    Generic,
    Literal,
    Self,
    TypeAlias,
    TypeVar,
    overload,
)

from tsdm.backend.generic import (
    false_like as universal_false_like,
    true_like as universal_true_like,
)
from tsdm.backend.numpy import numpy_apply_along_axes, numpy_like
from tsdm.backend.pandas import (
    pandas_clip,
    pandas_false_like,
    pandas_like,
    pandas_nanmax,
    pandas_nanmean,
    pandas_nanmin,
    pandas_nanstd,
    pandas_strip_whitespace,
    pandas_true_like,
    pandas_where,
)
from tsdm.backend.pyarrow import arrow_strip_whitespace
from tsdm.backend.torch import (
    torch_apply_along_axes,
    torch_like,
    torch_nanmax,
    torch_nanmin,
    torch_nanstd,
)
from tsdm.types.callback_protocols import (
    ApplyAlongAxes,
    ArraySplitProto,
    ClipProto,
    ConcatenateProto,
    ContractionProto,
    MakeScalarProto,
    SelfMap,
    TensorLikeProto,
    ToTensorProto,
    WhereProto,
)
from tsdm.types.protocols import SupportsArray
from tsdm.types.variables import T

BackendID: TypeAlias = Literal["arrow", "numpy", "pandas", "torch", "polars"]
r"""A type alias for the supported backends."""
BACKENDS = ("arrow", "numpy", "pandas", "torch", "polars")
r"""A tuple of the supported backends."""

Array = TypeVar("Array", bound=SupportsArray)
r"""TypeVar for tensor-like objects."""


def gather_types(obj: object, /) -> set[BackendID]:
    r"""Gather the backend types of a set of objects."""
    match obj:
        case (tuple() | set() | frozenset() | list()) as container:
            return set().union(*map(gather_types, container))
        case dict(mapping):
            return set().union(*map(gather_types, mapping.values()))
        case Tensor():
            return {"torch"}
        case pd.DataFrame() | pd.Series() | pd.Index():
            return {"pandas"}
        case pl.Series() | pl.DataFrame():
            return {"polars"}
        case pa.Array() | pa.Table():
            return {"arrow"}
        case ndarray():
            return {"numpy"}
        case (
            None
            | bool()
            | int()
            | float()
            | complex()
            | str()
            | datetime()
            | timedelta()
            | EllipsisType()
            | NotImplementedType()
        ):
            # FIXME: https://github.com/python/cpython/issues/106246
            # use PythonScalar instead of Scalar when the above issue is fixed
            return set()
        case _:
            raise TypeError(f"Unsupported type: {type(obj)}.")


def get_backend_id(obj: object, /, *, fallback: BackendID = "numpy") -> BackendID:
    r"""Get the backend of a set of objects."""
    types: set[BackendID] = gather_types(obj)

    match len(types):
        case 0:
            return fallback
        case 1:
            return types.pop()
        case _:
            raise ValueError(f"More than 1 backend detected: {types}.")


class Kernels:  # Q: how to make this more elegant?
    r"""A collection of kernels for numerical operations."""

    clip: Mapping[BackendID, ClipProto] = {
        "numpy": np.clip,
        "pandas": pandas_clip,
        "torch": torch.clip,
    }

    isnan: Mapping[BackendID, SelfMap] = {
        "numpy": np.isnan,
        "pandas": pd.isna,
        "torch": torch.isnan,
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

    false_like: Mapping[BackendID, SelfMap] = {
        "numpy": universal_false_like,
        "pandas": pandas_false_like,
        "torch": universal_false_like,
    }

    true_like: Mapping[BackendID, SelfMap] = {
        "numpy": universal_true_like,
        "pandas": pandas_true_like,
        "torch": universal_true_like,
    }

    tensor_like: Mapping[BackendID, TensorLikeProto] = {
        "numpy": numpy_like,
        "pandas": pandas_like,
        "torch": torch_like,
    }

    to_tensor: Mapping[BackendID, ToTensorProto] = {
        "numpy": np.array,
        "pandas": np.array,
        "torch": torch.tensor,
    }

    where: Mapping[BackendID, WhereProto] = {
        "numpy": np.where,
        "pandas": pandas_where,
        "torch": torch.where,  # type: ignore[dict-item]
    }

    strip_whitespace: Mapping[BackendID, SelfMap] = {
        "pandas": pandas_strip_whitespace,
        "arrow": arrow_strip_whitespace,
    }

    apply_along_axes: Mapping[BackendID, ApplyAlongAxes] = {
        "numpy": numpy_apply_along_axes,
        "torch": torch_apply_along_axes,
    }

    array_split: Mapping[BackendID, ArraySplitProto] = {
        "numpy": np.array_split,
        "torch": torch.tensor_split,  # type: ignore[dict-item]
    }

    concatenate: Mapping[BackendID, ConcatenateProto] = {
        "numpy": np.concatenate,
        "pandas": pd.concat,
        "torch": torch.cat,  # type: ignore[dict-item]
    }

    dropna: Mapping[BackendID, SelfMap] = {
        "arrow": lambda x: x.filter(x.is_valid()),
        "numpy": lambda x: x[~np.isnan(x)],
        "pandas": lambda x: x.dropna(),
        "polars": lambda x: x.drop_nulls(),
        "torch": lambda x: x[~torch.isnan(x)],
    }

    cast: Mapping[BackendID, SelfMap] = {
        "arrow": lambda x, dtype: x.cast(dtype),
        "numpy": lambda x, dtype: x.astype(dtype),
        "pandas": lambda x, dtype: x.astype(dtype),
        "polars": lambda x, dtype: x.cast(dtype),
        "torch": lambda x, dtype: x.to(dtype),
    }

    make_scalar: Mapping[BackendID, MakeScalarProto] = {
        "arrow": lambda value, dtype: pa.scalar(value, type=dtype),
        "numpy": lambda value, dtype: np.array([value], dtype=dtype),
        "pandas": lambda value, dtype: pd.Index([value]).astype(dtype).item(),
        "polars": lambda value, dtype: pl.Series([value]).cast(dtype).item(),
        "torch": lambda value, dtype: torch.tensor(value, dtype=dtype),
    }


@dataclass(frozen=True, slots=True, init=False)
class Backend(Generic[T]):
    r"""Provides kernels for numerical operations."""

    # __slots__ = ("selected_backend", *Kernels.__annotations__.keys())

    NAME: Final[BackendID]

    # KERNELS
    clip: ClipProto[T]
    isnan: SelfMap[T]
    where: WhereProto[T]

    # nan-aggregations
    nanmax: ContractionProto[T]
    nanmean: ContractionProto[T]
    nanmin: ContractionProto[T]
    nanstd: ContractionProto[T]

    tensor_like: TensorLikeProto[T]
    to_tensor: ToTensorProto[T]
    true_like: SelfMap[T]
    false_like: SelfMap[T]

    strip_whitespace: SelfMap[T]
    apply_along_axes: ApplyAlongAxes[T]

    array_split: ArraySplitProto[T]
    concatenate: ConcatenateProto[T]

    dropna: SelfMap[T]
    cast: SelfMap[T]
    make_scalar: MakeScalarProto

    def __init__(self, backend: str | Self) -> None:
        # set the selected backend
        name = backend.NAME if isinstance(backend, Backend) else str(backend)

        if name not in BACKENDS:
            raise ValueError(f"Invalid backend: {name}.")

        # NOTE: Need to use object.__setattr__ for frozen dataclasses.
        object.__setattr__(self, "NAME", name)

        for attr in Kernels.__annotations__:
            implementations = getattr(Kernels, attr)
            impl = implementations.get(name, NotImplemented)
            object.__setattr__(self, attr, impl)


@overload
def get_backend(obj: Array, /, *, fallback: BackendID = ...) -> Backend[Array]: ...
@overload
def get_backend(obj: object, /, *, fallback: BackendID = ...) -> Backend: ...
def get_backend(obj: object, /, *, fallback: BackendID = "numpy") -> Backend:
    r"""Get the backend of a set of objects."""
    backend_id = get_backend_id(obj, fallback=fallback)
    return Backend(backend_id)
