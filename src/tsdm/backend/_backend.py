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

import numpy
import pandas
import torch
from numpy import ndarray
from pandas import DataFrame, Series
from torch import Tensor
from typing_extensions import (
    Generic,
    Literal,
    ParamSpec,
    Self,
    TypeAlias,
    cast,
    get_args,
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
from tsdm.backend.universal import (
    false_like as universal_false_like,
    true_like as universal_true_like,
)
from tsdm.types.callback_protocols import (
    ApplyAlongAxes,
    ClipProto,
    ContractionProto,
    SelfMap,
    TensorLikeProto,
    ToTensorProto,
    WhereProto,
)
from tsdm.types.variables import any_var as T

P = ParamSpec("P")

BackendID: TypeAlias = Literal["arrow", "numpy", "pandas", "torch"]
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
        case DataFrame() | Series():
            types.add("pandas")
        case ndarray():
            types.add("numpy")
        case (
            None
            | bool()
            | int()
            | float()
            | complex()
            | str()
            | datetime()
            | timedelta()
        ):
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


def get_backend(obj: object, /, *, fallback: BackendID = "numpy") -> BackendID:
    """Get the backend of a set of objects."""
    types: set[BackendID] = gather_types(obj)

    match len(types):
        case 0:
            return fallback
        case 1:
            return types.pop()
        case _:
            raise ValueError(f"More than 1 backend detected: {types}.")


class Kernels:  # Q: how to make this more elegant?
    """A collection of kernels for numerical operations."""

    clip: Mapping[BackendID, ClipProto] = {
        "numpy": numpy.clip,
        "pandas": pandas_clip,
        "torch": torch.clip,
    }

    isnan: Mapping[BackendID, SelfMap] = {
        "numpy": numpy.isnan,
        "pandas": pandas.isna,
        "torch": torch.isnan,
    }

    nanmin: Mapping[BackendID, ContractionProto] = {
        "numpy": numpy.nanmin,
        "pandas": pandas_nanmin,
        "torch": torch_nanmin,
    }

    nanmax: Mapping[BackendID, ContractionProto] = {
        "numpy": numpy.nanmax,
        "pandas": pandas_nanmax,
        "torch": torch_nanmax,
    }

    nanmean: Mapping[BackendID, ContractionProto] = {
        "numpy": numpy.nanmean,
        "pandas": pandas_nanmean,
        "torch": torch.nanmean,  # type: ignore[dict-item]
    }

    nanstd: Mapping[BackendID, ContractionProto] = {
        "numpy": numpy.nanstd,
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
        "numpy": numpy.array,
        "pandas": numpy.array,
        "torch": torch.tensor,
    }

    where: Mapping[BackendID, WhereProto] = {
        "numpy": numpy.where,
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


class Backend(Generic[T]):
    """Provides kernels for numerical operations."""

    __slots__ = ("selected_backend", *Kernels.__annotations__.keys())

    selected_backend: BackendID

    # KERNELS
    clip: ClipProto[T]
    isnan: SelfMap[T]
    where: WhereProto[T]

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

    def __init__(self, backend: str | Self) -> None:
        # set the selected backend
        if isinstance(backend, Backend):
            backend = backend.selected_backend

        assert backend in get_args(BackendID)
        self.selected_backend = cast(BackendID, backend)

        for attr in Kernels.__annotations__:
            implementations = getattr(Kernels, attr)
            impl = implementations.get(self.selected_backend, NotImplemented)
            setattr(self, attr, impl)

        # # select the kernels
        # self.clip = Kernels.clip.get(self.selected_backend, NotImplemented)
        # self.isnan = Kernels.isnan.get(self.selected_backend, NotImplemented)
        # self.where = Kernels.where.get(self.selected_backend, NotImplemented)
        #
        # # contractions
        # self.nanmax = Kernels.nanmax.get(self.selected_backend, NotImplemented)
        # self.nanmean = Kernels.nanmean.get(self.selected_backend, NotImplemented)
        # self.nanmin = Kernels.nanmin.get(self.selected_backend, NotImplemented)
        # self.nanstd = Kernels.nanstd.get(self.selected_backend, NotImplemented)
        #
        # # tensor creation
        # self.tensor_like = Kernels.tensor_like.get(
        #     self.selected_backend, NotImplemented
        # )
        # self.to_tensor = Kernels.to_tensor.get(self.selected_backend, NotImplemented)
        # self.true_like = Kernels.true_like.get(self.selected_backend, NotImplemented)
        # self.false_like = Kernels.false_like.get(self.selected_backend, NotImplemented)
        #
        # # other
        # self.strip_whitespace = Kernels.strip_whitespace.get(
        #     self.selected_backend, NotImplemented
        # )
        # self.apply_along_axes = Kernels.apply_along_axes.get(
        #     self.selected_backend, NotImplemented
        # )
