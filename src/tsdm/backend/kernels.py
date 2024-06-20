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
import pyarrow.compute as pc
import torch as pt
from numpy import ndarray
from torch import Tensor
from typing_extensions import Generic, Literal, Self, TypeAlias, TypeVar, overload

from tsdm import backend
from tsdm.types.callback_protocols import (
    ApplyAlongAxes,
    ArraySplitProto,
    CastProto,
    ClipProto,
    ConcatenateProto,
    ContractionProto,
    CopyLikeProto,
    FullLikeProto,
    ScalarProto,
    SelfMap,
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
        "pandas": backend.pandas.clip,
        "torch": pt.clip,
    }

    is_null: Mapping[BackendID, SelfMap] = {
        "arrow": pc.is_null,
        "numpy": np.isnan,
        "pandas": pd.isna,
        "polars": backend.polars.is_null,
        "torch": pt.isnan,
    }

    nanmin: Mapping[BackendID, ContractionProto] = {
        "numpy": np.nanmin,
        "pandas": backend.pandas.nanmin,
        "polars": backend.polars.nanmin,
        "torch": backend.torch.nanmin,
    }

    nanmax: Mapping[BackendID, ContractionProto] = {
        "numpy": np.nanmax,
        "pandas": backend.pandas.nanmax,
        "polars": backend.polars.nanmax,
        "torch": backend.torch.nanmax,
    }

    nanmean: Mapping[BackendID, ContractionProto] = {
        "numpy": np.nanmean,
        "pandas": backend.pandas.nanmean,
        "torch": pt.nanmean,  # type: ignore[dict-item]
    }

    nanstd: Mapping[BackendID, ContractionProto] = {
        "numpy": np.nanstd,
        "pandas": backend.pandas.nanstd,
        "torch": backend.torch.nanstd,
    }

    false_like: Mapping[BackendID, SelfMap] = {
        "arrow": backend.pyarrow.false_like,
        "numpy": backend.generic.false_like,
        "pandas": backend.pandas.false_like,
        "torch": backend.generic.false_like,
    }

    true_like: Mapping[BackendID, SelfMap] = {
        "arrow": backend.pyarrow.true_like,
        "numpy": backend.generic.true_like,
        "pandas": backend.pandas.true_like,
        "torch": backend.generic.true_like,
    }

    null_like: Mapping[BackendID, SelfMap] = {
        "arrow": backend.pyarrow.null_like,
        "pandas": backend.pandas.null_like,
    }

    full_like: Mapping[BackendID, FullLikeProto] = {
        "arrow": backend.pyarrow.full_like,
        "numpy": np.full_like,
    }

    copy_like: Mapping[BackendID, CopyLikeProto] = {
        "numpy": backend.numpy.copy_like,
        "pandas": backend.pandas.copy_like,
        "torch": backend.torch.copy_like,
    }

    to_tensor: Mapping[BackendID, ToTensorProto] = {
        "numpy": np.array,
        "pandas": np.array,
        "torch": pt.tensor,
    }

    where: Mapping[BackendID, WhereProto] = {
        "arrow": backend.pyarrow.where,
        "numpy": np.where,
        "pandas": backend.pandas.where,
        "torch": pt.where,  # type: ignore[dict-item]
    }

    strip_whitespace: Mapping[BackendID, SelfMap] = {
        "pandas": backend.pandas.strip_whitespace,
        "arrow": backend.pyarrow.strip_whitespace,
    }

    apply_along_axes: Mapping[BackendID, ApplyAlongAxes] = {
        "numpy": backend.numpy.apply_along_axes,
        "torch": backend.torch.apply_along_axes,
    }

    array_split: Mapping[BackendID, ArraySplitProto] = {
        "numpy": np.array_split,
        "torch": pt.tensor_split,  # type: ignore[dict-item]
    }

    concatenate: Mapping[BackendID, ConcatenateProto] = {
        "numpy": np.concatenate,
        "pandas": pd.concat,
        "torch": pt.cat,  # type: ignore[dict-item]
    }

    drop_null: Mapping[BackendID, SelfMap] = {
        "arrow": pc.drop_null,
        "numpy": backend.numpy.drop_null,
        "pandas": backend.pandas.drop_null,
        "polars": backend.polars.drop_null,
        "torch": backend.torch.drop_null,
    }

    cast: Mapping[BackendID, CastProto] = {
        "arrow": pc.cast,
        "numpy": ndarray.astype,
        "pandas": backend.pandas.cast,
        "polars": backend.polars.cast,
        "torch": Tensor.to,
    }

    scalar: Mapping[BackendID, ScalarProto] = {
        "arrow": backend.pyarrow.scalar,
        "numpy": backend.numpy.scalar,
        "pandas": backend.pandas.scalar,
        "polars": backend.polars.scalar,
        "torch": backend.torch.scalar,
    }


@dataclass(frozen=True, slots=True, init=False)
class Backend(Generic[T]):
    r"""Provides kernels for numerical operations."""

    # __slots__ = ("selected_backend", *Kernels.__annotations__.keys())

    NAME: BackendID

    # KERNELS
    clip: ClipProto[T]
    drop_null: SelfMap[T]
    is_null: SelfMap[T]
    where: WhereProto[T]

    # nan-aggregations
    nanmax: ContractionProto[T]
    nanmean: ContractionProto[T]
    nanmin: ContractionProto[T]
    nanstd: ContractionProto[T]

    # array creation
    copy_like: CopyLikeProto[T]
    false_like: SelfMap[T]
    full_like: FullLikeProto[T]
    null_like: SelfMap[T]
    true_like: SelfMap[T]

    # string operations
    strip_whitespace: SelfMap[T]
    apply_along_axes: ApplyAlongAxes[T]

    # array construction
    array_split: ArraySplitProto[T]
    concatenate: ConcatenateProto[T]

    cast: CastProto[T]
    scalar: ScalarProto
    to_tensor: ToTensorProto[T]

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
