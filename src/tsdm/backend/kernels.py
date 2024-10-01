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

from dataclasses import dataclass
from datetime import datetime, timedelta
from types import EllipsisType, NotImplementedType
from typing import Literal, Self, overload

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import torch as pt
from numpy import ndarray
from torch import Tensor

from tsdm import backend as B
from tsdm.types.arrays import SupportsArray
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

type BackendID = Literal["arrow", "numpy", "pandas", "torch", "polars"]
r"""A type alias for the supported backends."""
BACKENDS = ("arrow", "numpy", "pandas", "torch", "polars")
r"""A tuple of the supported backends."""


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

    clip: dict[BackendID, ClipProto] = {
        "numpy": np.clip,
        "pandas": B.pandas.clip,
        "torch": pt.clip,
    }

    is_null: dict[BackendID, SelfMap] = {
        "arrow": pc.is_null,
        "numpy": np.isnan,
        "pandas": pd.isna,
        "polars": B.polars.is_null,
        "torch": pt.isnan,
    }

    nanmin: dict[BackendID, ContractionProto] = {
        "numpy": np.nanmin,
        "pandas": B.pandas.nanmin,
        "polars": B.polars.nanmin,
        "torch": B.torch.nanmin,
    }

    nanmax: dict[BackendID, ContractionProto] = {
        "numpy": np.nanmax,
        "pandas": B.pandas.nanmax,
        "polars": B.polars.nanmax,
        "torch": B.torch.nanmax,
    }

    nanmean: dict[BackendID, ContractionProto] = {
        "numpy": np.nanmean,
        "pandas": B.pandas.nanmean,
        "torch": pt.nanmean,  # type: ignore[dict-item]
    }

    nanstd: dict[BackendID, ContractionProto] = {
        "numpy": np.nanstd,
        "pandas": B.pandas.nanstd,
        "torch": B.torch.nanstd,
    }

    false_like: dict[BackendID, SelfMap] = {
        "arrow": B.pyarrow.false_like,
        "numpy": B.generic.false_like,
        "pandas": B.pandas.false_like,
        "torch": B.generic.false_like,
    }

    true_like: dict[BackendID, SelfMap] = {
        "arrow": B.pyarrow.true_like,
        "numpy": B.generic.true_like,
        "pandas": B.pandas.true_like,
        "torch": B.generic.true_like,
    }

    null_like: dict[BackendID, SelfMap] = {
        "arrow": B.pyarrow.null_like,
        "pandas": B.pandas.null_like,
    }

    full_like: dict[BackendID, FullLikeProto] = {
        "arrow": B.pyarrow.full_like,
        "numpy": np.full_like,
    }

    copy_like: dict[BackendID, CopyLikeProto] = {
        "numpy": B.numpy.copy_like,
        "pandas": B.pandas.copy_like,
        "torch": B.torch.copy_like,
    }

    to_tensor: dict[BackendID, ToTensorProto] = {
        "numpy": np.array,
        "pandas": np.array,
        "torch": pt.tensor,
    }

    where: dict[BackendID, WhereProto] = {
        "arrow": B.pyarrow.where,
        "numpy": np.where,
        "pandas": B.pandas.where,
        "torch": pt.where,  # type: ignore[dict-item]
    }

    strip_whitespace: dict[BackendID, SelfMap] = {
        "pandas": B.pandas.strip_whitespace,
        "arrow": B.pyarrow.strip_whitespace,
    }

    apply_along_axes: dict[BackendID, ApplyAlongAxes] = {
        "numpy": B.numpy.apply_along_axes,
        "torch": B.torch.apply_along_axes,
    }

    array_split: dict[BackendID, ArraySplitProto] = {
        "numpy": np.array_split,
        "torch": pt.tensor_split,  # type: ignore[dict-item]
    }

    concatenate: dict[BackendID, ConcatenateProto] = {
        "numpy": np.concatenate,
        "pandas": pd.concat,
        "torch": pt.cat,  # type: ignore[dict-item]
    }

    drop_null: dict[BackendID, SelfMap] = {
        "arrow": pc.drop_null,
        "numpy": B.numpy.drop_null,
        "pandas": B.pandas.drop_null,
        "polars": B.polars.drop_null,
        "torch": B.torch.drop_null,
    }

    cast: dict[BackendID, CastProto] = {
        "arrow": pc.cast,
        "numpy": ndarray.astype,
        "pandas": B.pandas.cast,
        "polars": B.polars.cast,
        "torch": Tensor.to,
    }

    scalar: dict[BackendID, ScalarProto] = {
        "arrow": B.pyarrow.scalar,
        "numpy": B.numpy.scalar,
        "pandas": B.pandas.scalar,
        "polars": B.polars.scalar,
        "torch": B.torch.scalar,
    }


@dataclass(frozen=True, slots=True, init=False)
class Backend[T]:
    r"""Provides kernels for numerical operations."""

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
def get_backend[Arr: SupportsArray](
    obj: Arr, /, *, fallback: BackendID = ...
) -> Backend[Arr]: ...
@overload
def get_backend(obj: object, /, *, fallback: BackendID = ...) -> Backend: ...
def get_backend(obj: object, /, *, fallback: BackendID = "numpy") -> Backend:
    r"""Get the backend of a set of objects."""
    backend_id = get_backend_id(obj, fallback=fallback)
    return Backend(backend_id)
