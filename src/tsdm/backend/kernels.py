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
]

from dataclasses import dataclass
from datetime import datetime, timedelta
from types import EllipsisType, NotImplementedType
from typing import Literal, Self, cast, overload

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import torch as pt
from numpy import ndarray
from torch import Tensor

from tsdm import constants
from tsdm.types import SupportsArray

from . import (
    generic,
    numpy as numpy_backend,
    pandas as pandas_backend,
    polars as polars_backend,
    pyarrow as pyarrow_backend,
    torch as torch_backend,
)
from .schemas import (
    ApplyAlongAxes,
    ArraySplitProto,
    CastProto,
    ClipProto,
    ConcatenateProto,
    ContractionProto,
    CopyLikeProto,
    FullLikeProto,
    RemoveOutliersProto,
    ScalarProto,
    SelectOutliersProto,
    SelfMap,
    StripWhitespaceProto,
    ToTensorProto,
    WhereProto,
)

type BackendID = Literal["generic", "arrow", "numpy", "pandas", "polars", "torch"]
r"""A type alias for the supported backends."""
BACKENDS = ("generic", "arrow", "numpy", "pandas", "polars", "torch")
r"""A tuple of the supported backends."""


class Kernels:  # TODO: how to make this more elegant?
    r"""A collection of kernels for numerical operations."""

    select_outliers: dict[BackendID, SelectOutliersProto] = {
        "arrow": pyarrow_backend.select_outliers,
        "pandas": pandas_backend.select_outliers,
        "polars": polars_backend.select_outliers,
    }

    remove_outliers: dict[BackendID, RemoveOutliersProto] = {
        "arrow": pyarrow_backend.remove_outliers,
        "pandas": pandas_backend.remove_outliers,
        "polars": polars_backend.remove_outliers,
    }

    clip: dict[BackendID, ClipProto] = {
        "numpy": np.clip,
        "pandas": pandas_backend.clip,
        "torch": pt.clip,
    }

    is_null: dict[BackendID, SelfMap] = {
        "arrow": pc.is_null,
        "numpy": np.isnan,
        "pandas": pd.isna,
        "polars": polars_backend.is_null,
        "torch": pt.isnan,
    }

    nanmin: dict[BackendID, ContractionProto] = {  # ty: ignore[invalid-assignment]
        "numpy": np.nanmin,  # pyrefly: ignore[bad-assignment]
        "pandas": pandas_backend.nanmin,
        "polars": polars_backend.nanmin,
        "torch": torch_backend.nanmin,
    }

    nanmax: dict[BackendID, ContractionProto] = {  # ty: ignore[invalid-assignment]
        "numpy": np.nanmax,  # pyrefly: ignore[bad-assignment]
        "pandas": pandas_backend.nanmax,
        "polars": polars_backend.nanmax,
        "torch": torch_backend.nanmax,
    }

    nanmean: dict[BackendID, ContractionProto] = {  # ty: ignore[invalid-assignment]
        "numpy": np.nanmean,  # pyrefly: ignore[bad-assignment]
        "pandas": pandas_backend.nanmean,
        "torch": cast("ContractionProto[Tensor]", pt.nanmean),
    }

    nanstd: dict[BackendID, ContractionProto] = {  # ty: ignore[invalid-assignment]
        "numpy": np.nanstd,  # pyrefly: ignore[bad-assignment]
        "pandas": pandas_backend.nanstd,
        "torch": torch_backend.nanstd,
    }

    false_like: dict[BackendID, SelfMap] = {
        "generic": generic.false_like,
        "arrow": pyarrow_backend.false_like,
        "pandas": pandas_backend.false_like,
    }

    true_like: dict[BackendID, SelfMap] = {
        "generic": generic.true_like,
        "arrow": pyarrow_backend.true_like,
        "pandas": pandas_backend.true_like,
    }

    null_like: dict[BackendID, SelfMap] = {
        "arrow": pyarrow_backend.null_like,
        "pandas": pandas_backend.null_like,
    }

    full_like: dict[BackendID, FullLikeProto] = {
        "arrow": pyarrow_backend.full_like,
        "numpy": np.full_like,
    }

    copy_like: dict[BackendID, CopyLikeProto] = {
        "numpy": numpy_backend.copy_like,
        "pandas": pandas_backend.copy_like,
        "torch": torch_backend.copy_like,
    }

    to_tensor: dict[BackendID, ToTensorProto] = {
        "numpy": np.array,
        "pandas": np.array,
        "torch": pt.tensor,
    }

    where: dict[BackendID, WhereProto] = {
        "generic": generic.where,
        "arrow": pyarrow_backend.where,
        "numpy": np.where,
        "pandas": pandas_backend.where,
        "torch": pt.where,
    }

    strip_whitespace: dict[BackendID, StripWhitespaceProto] = {
        "pandas": pandas_backend.strip_whitespace,
        "polars": polars_backend.strip_whitespace,
        "arrow": pyarrow_backend.strip_whitespace,
    }

    apply_along_axes: dict[BackendID, ApplyAlongAxes] = {
        "numpy": numpy_backend.apply_along_axes,
        "torch": torch_backend.apply_along_axes,
    }

    array_split: dict[BackendID, ArraySplitProto] = {
        "numpy": np.array_split,
        "torch": cast("ArraySplitProto[Tensor]", pt.tensor_split),
    }

    concatenate: dict[BackendID, ConcatenateProto] = {
        "numpy": np.concatenate,
        "pandas": pd.concat,
        "torch": cast("ConcatenateProto[Tensor]", pt.cat),
    }

    drop_null: dict[BackendID, SelfMap] = {
        "arrow": pc.drop_null,
        "numpy": numpy_backend.drop_null,
        "pandas": pandas_backend.drop_null,
        "polars": polars_backend.drop_null,
        "torch": torch_backend.drop_null,
    }

    cast: dict[BackendID, CastProto] = {
        "arrow": pc.cast,
        "numpy": ndarray.astype,
        "pandas": pandas_backend.cast,
        "polars": polars_backend.cast,
        "torch": Tensor.to,
    }

    scalar: dict[BackendID, ScalarProto] = {
        "arrow": pyarrow_backend.scalar,
        "numpy": numpy_backend.scalar,
        "pandas": pandas_backend.scalar,
        "polars": polars_backend.scalar,
        "torch": torch_backend.scalar,
    }


@dataclass(slots=True, frozen=True, init=False)
class Backend[T]:
    r"""Provide numerical kernels for a selected backend.

    Kernels without a backend-specific implementation use a generic implementation
    when available. Unsupported kernels are set to `None`.
    """

    NAME: BackendID

    # KERNELS
    select_outliers: SelectOutliersProto[T, T]
    remove_outliers: RemoveOutliersProto[T, T]
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
    strip_whitespace: StripWhitespaceProto[T, T]
    apply_along_axes: ApplyAlongAxes[T]

    # array construction
    array_split: ArraySplitProto[T]
    concatenate: ConcatenateProto[T]

    cast: CastProto[T]
    scalar: ScalarProto
    to_tensor: ToTensorProto[T]

    def __init__(self, backend: str | Self) -> None:
        # set the selected backend
        name = backend.NAME if isinstance(backend, Backend) else backend

        if name not in BACKENDS:
            raise ValueError(f"Invalid backend: {name}.")

        # NOTE: Need to use object.__setattr__ for frozen dataclasses.
        object.__setattr__(self, "NAME", name)

        for attr in Kernels.__annotations__:
            implementations = getattr(Kernels, attr)
            impl = implementations.get(name, None)
            if impl is None:  # fallback to generic kernel
                impl = implementations.get("generic", None)
            object.__setattr__(self, attr, impl)


def _gather_types(obj: object, /) -> set[BackendID]:
    r"""Gather backend identifiers from an object or nested containers.

    Inspect dictionary values and elements of lists, tuples, sets, and frozensets
    recursively. Python scalars and sentinel values contribute no identifiers.
    Unsupported objects raise `TypeError`.
    """
    match obj:
        case (tuple() | set() | frozenset() | list()) as container:
            return {*_gather_types(item) for item in container}  # pyrefly: ignore[invalid-annotation, bad-return]
        case dict(mapping):
            return {*_gather_types(item) for item in mapping.values()}  # pyrefly: ignore[invalid-annotation, bad-return]
        case Tensor():
            return {"torch"}
        case pd.DataFrame() | pd.Series() | pd.Index():
            return {"pandas"}
        case pl.Series() | pl.DataFrame():
            return {"polars"}
        case pa.Array() | pa.ChunkedArray() | pa.Table():
            return {"arrow"}
        case ndarray():
            return {"numpy"}
        case (
            constants.UNDEFINED
            | None
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
    r"""Return the backend identifier of an object or nested containers.

    Args:
        obj: Object or nested containers to inspect.
        fallback: Backend identifier to return when no backend is detected.

    Returns:
        The detected backend identifier, or `fallback` for scalars and empty containers.

    Raises:
        TypeError: If an object has an unsupported type.
        ValueError: If more than one backend is detected.
    """
    types: set[BackendID] = _gather_types(obj)

    match len(types):
        case 0:
            return fallback
        case 1:
            return types.pop()
        case _:
            raise ValueError(f"More than 1 backend detected: {types}.")


@overload
def get_backend[Arr: SupportsArray](
    obj: Arr, /, *, fallback: BackendID = ...
) -> Backend[Arr]: ...
@overload
def get_backend(obj: object, /, *, fallback: BackendID = ...) -> Backend: ...
def get_backend(obj: object, /, *, fallback: BackendID = "numpy") -> Backend:
    r"""Return numerical kernels for an object or nested containers.

    Args:
        obj: Object or nested containers to inspect.
        fallback: Backend to use when no backend is detected.

    Returns:
        A backend providing kernels for the detected type, or the fallback backend.

    Raises:
        TypeError: If an object has an unsupported type.
        ValueError: If multiple backends are detected or the backend name is invalid.
    """
    backend_id = get_backend_id(obj, fallback=fallback)
    return Backend(backend_id)
