"""Numerical functions with shared signatures for multiple backends."""

__all__ = [
    # Classes
    "BackendID",
    "Kernels",
    "Backend",
    # Functions
    "get_backend",
]

from collections.abc import Mapping
from datetime import datetime, timedelta
from types import EllipsisType, NotImplementedType
from typing import Generic, Literal, ParamSpec, TypeAlias, cast

import numpy
import numpy as np
import torch
from numpy import ndarray
from pandas import DataFrame, Series
from torch import Tensor
from typing_extensions import get_args

from tsdm.backend.numpy import numpy_like
from tsdm.backend.pandas import (
    pandas_clip,
    pandas_like,
    pandas_nanmax,
    pandas_nanmean,
    pandas_nanmin,
    pandas_nanstd,
    pandas_where,
)
from tsdm.backend.protocols import (
    ClipProto,
    ContractionProto,
    TensorLikeProto,
    ToTensorProto,
    WhereProto,
)
from tsdm.backend.torch import torch_like, torch_nanmax, torch_nanmin, torch_nanstd
from tsdm.backend.universal import (
    false_like as universal_false_like,
    true_like as universal_true_like,
)
from tsdm.types.protocols import SelfMapProto
from tsdm.types.variables import any_var as T

P = ParamSpec("P")

BackendID: TypeAlias = Literal["torch", "numpy", "pandas"]
"""A type alias for the supported backends."""


def gather_types(obj: object) -> set[BackendID]:
    """Gather the backend types of a set of objects."""
    types: set[BackendID] = set()

    match obj:
        case tuple() | set() | frozenset() | list() as container:
            types |= set().union(*map(gather_types, container))
        case dict() as mapping:
            types |= set().union(*map(gather_types, mapping.values()))
        case Tensor():
            types.add("torch")
        case DataFrame() | Series():  # type: ignore[misc]
            types.add("pandas")  # type: ignore[unreachable]
        case ndarray():
            types.add("numpy")
        case None | bool() | int() | float() | complex() | str() | datetime() | timedelta():
            # FIXME: https://github.com/python/cpython/issues/106246
            # use PythonScalar instead of Scalar when the above issue is fixed
            # types.add("fallback")
            pass
        case EllipsisType() | NotImplementedType():
            # types.add("fallback")
            pass
        case _:
            raise TypeError(f"Unsupported type: {type(obj)}.")

    return types


def get_backend(obj: object, fallback: BackendID = "numpy") -> BackendID:
    """Get the backend of a set of objects."""
    types: set[BackendID] = gather_types(obj)

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

    tensor_like: Mapping[BackendID, TensorLikeProto] = {
        "numpy": numpy_like,
        "pandas": pandas_like,
        "torch": torch_like,
    }

    to_tensor: Mapping[BackendID, ToTensorProto] = {
        "numpy": numpy.array,
        "pandas": numpy.array,
        "torch": torch.tensor,
    }

    where: Mapping[BackendID, WhereProto] = {
        "numpy": numpy.where,
        "pandas": pandas_where,
        "torch": torch.where,  # type: ignore[dict-item]
    }


class Backend(Generic[T]):
    """Provides kernels for numerical operations."""

    selected_backend: BackendID

    # KERNELS
    clip: ClipProto[T]
    where: WhereProto[T]

    nanmax: ContractionProto[T]
    nanmean: ContractionProto[T]
    nanmin: ContractionProto[T]
    nanstd: ContractionProto[T]

    tensor_like: TensorLikeProto[T]
    to_tensor: ToTensorProto[T]
    true_like: SelfMapProto[T]
    false_like: SelfMapProto[T]

    def __init__(self, backend: str) -> None:
        # set the selected backend
        assert backend in get_args(BackendID)
        self.selected_backend = cast(BackendID, backend)

        # select the kernels
        self.clip = Kernels.clip[self.selected_backend]
        self.where = Kernels.where[self.selected_backend]

        # contractions
        self.nanmax = Kernels.nanmax[self.selected_backend]
        self.nanmean = Kernels.nanmean[self.selected_backend]
        self.nanmin = Kernels.nanmin[self.selected_backend]
        self.nanstd = Kernels.nanstd[self.selected_backend]

        # tensor creation
        self.tensor_like = Kernels.tensor_like[self.selected_backend]
        self.to_tensor = Kernels.to_tensor[self.selected_backend]
        self.true_like = Kernels.true_like[self.selected_backend]
        self.false_like = Kernels.false_like[self.selected_backend]
