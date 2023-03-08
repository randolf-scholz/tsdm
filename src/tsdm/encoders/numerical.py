r"""Numerical Transformations, Standardization, Log-Transforms, etc."""

from __future__ import annotations

__all__ = [
    # Classes
    "BoundaryEncoder",
    "LinearScaler",
    "LogEncoder",
    "LogitEncoder",
    "MinMaxScaler",
    "Standardizer",
    "TensorConcatenator",
    "TensorSplitter",
]

from dataclasses import KW_ONLY, dataclass
from typing import (
    Any,
    Generic,
    Literal,
    NamedTuple,
    Optional,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pandas import NA, DataFrame, Index, Series
from torch import Tensor

from tsdm.encoders.base import BaseEncoder
from tsdm.types.aliases import PandasObject
from tsdm.utils.strings import repr_namedtuple

TensorLike: TypeAlias = Tensor | NDArray | DataFrame | Series
r"""Type Hint for tensor-like objects."""
TensorType = TypeVar("TensorType", Tensor, np.ndarray, DataFrame, Series)
r"""TypeVar for tensor-like objects."""


def get_broadcast(
    data: Any, /, *, axis: tuple[int, ...] | None
) -> tuple[slice | None, ...]:
    r"""Create an indexer for axis specific broadcasting.

    Example:
        data is of shape  ``(2,3,4,5,6,7)``
        axis is the tuple ``(0,2,-1)``
        broadcast is ``(:, None, :, None, None, :)``

    Then, given a tensor ``x`` of shape ``(2, 4, 7)``, we can perform element-wise
    operations via ``data + x[broadcast]``.
    """
    rank = len(data.shape)

    if axis is None:
        return (slice(None),) * rank

    axis = tuple(a % rank for a in axis)
    broadcast = tuple(slice(None) if a in axis else None for a in range(rank))
    return broadcast


@dataclass
class BoundaryEncoder(BaseEncoder):
    r"""Clip or mask values outside a given range.

    If `mode='mask'`, then values outside the boundary will be replaced by `NA`.
    If `mode='clip'`, then values outside the boundary will be clipped to it.
    """

    lower: float | np.ndarray
    upper: float | np.ndarray

    _: KW_ONLY

    axis: int | tuple[int, ...] = -1
    mode: Literal["mask", "clip"] = "mask"
    mask_value: float = float("nan")

    requires_fit: bool = False

    def __post_init__(self):
        self.lower_mask: float | np.ndarray
        self.upper_mask: float | np.ndarray
        if self.mode == "mask":
            self.lower_mask = self.mask_value
            self.upper_mask = self.mask_value
        elif self.mode == "clip":
            self.lower_mask = self.lower
            self.upper_mask = self.upper
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def fit(self, data: DataFrame) -> None:
        # TODO: make _nan adapt to real data type!
        if isinstance(data, Series | DataFrame) and pd.isna(self.mask_value):
            self.mask_value = NA

    def encode(self, data: DataFrame) -> DataFrame:
        # NOTE: frame.where(cond, other) replaces where condition is False!
        data = data.where(data.isna() | (data >= self.lower), self.lower_mask)
        data = data.where(data.isna() | (data <= self.upper), self.upper_mask)
        return data

    def decode(self, data: DataFrame) -> DataFrame:
        return data


class Standardizer(BaseEncoder, Generic[TensorType]):
    r"""A StandardScalar that works with batch dims."""

    mean: Any
    r"""The mean value."""
    stdv: Any
    r"""The standard-deviation."""
    ignore_nan: bool = True
    r"""Whether to ignore nan-values while fitting."""
    axis: tuple[int, ...]
    r"""The axis to perform the scaling. If None, automatically select the axis."""

    class Parameters(NamedTuple):
        r"""The parameters of the StandardScalar."""

        mean: TensorLike
        stdv: TensorLike
        axis: None | tuple[int, ...]

        def __repr__(self) -> str:
            r"""Pretty print."""
            return repr_namedtuple(self)

    def __init__(
        self,
        /,
        mean: Optional[Tensor] = None,
        stdv: Optional[Tensor] = None,
        *,
        ignore_nan: bool = True,
        axis: None | int | tuple[int, ...] = None,
    ):
        super().__init__()
        self.ignore_nan = ignore_nan
        self.axis = (axis,) if isinstance(axis, int) else axis  # type: ignore[assignment]
        self.mean = mean
        self.stdv = stdv

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(axis={self.axis})"

    def __getitem__(self, item: Any) -> Standardizer:
        r"""Return a slice of the Standardizer."""
        axis = self.axis if self.axis is None else self.axis[1:]
        encoder = Standardizer(mean=self.mean[item], stdv=self.stdv[item], axis=axis)
        encoder._is_fitted = self._is_fitted
        return encoder

    @property
    def param(self) -> Parameters:
        r"""Parameters of the Standardizer."""
        return self.Parameters(self.mean, self.stdv, self.axis)

    def fit(self, data: TensorType, /) -> None:
        rank = len(data.shape)

        if self.axis is None:
            self.axis = () if rank == 1 else (-1,)

        selection = [(a % rank) for a in self.axis]
        axes = tuple(k for k in range(rank) if k not in selection)

        if isinstance(data, Tensor):
            if self.ignore_nan:
                self.mean = torch.nanmean(data, dim=axes)
                mean = torch.nanmean(data, dim=axes, keepdim=True)
                self.stdv = torch.sqrt(torch.nanmean((data - mean) ** 2, dim=axes))
            else:
                self.mean = torch.mean(data, dim=axes)
                self.stdv = torch.std(data, dim=axes)
        else:
            if self.ignore_nan:
                self.mean = np.nanmean(data, axis=axes)
                self.stdv = np.nanstd(data, axis=axes)
            else:
                self.mean = np.mean(data, axis=axes)
                self.stdv = np.std(data, axis=axes)

        # if isinstance(data, Tensor):
        #     mask = ~torch.isnan(data) if self.ignore_nan else torch.full_like(data, True)
        #     count = torch.maximum(torch.tensor(1), mask.sum(dim=axes))
        #     masked = torch.where(mask, data, torch.tensor(0.0))
        #     self.mean = masked.sum(dim=axes)/count
        #     residual = apply_along_axes(masked, -self.mean, torch.add, axes=axes)
        #     self.stdv = torch.sqrt((residual**2).sum(dim=axes)/count)
        # else:
        #     if isinstance(data, DataFrame) or isinstance(data, Series):
        #         data = data.values
        #     mask = ~np.isnan(data) if self.ignore_nan else np.full_like(data, True)
        #     count = np.maximum(1, mask.sum(axis=axes))
        #     masked = np.where(mask, data, 0)
        #     self.mean = masked.sum(axis=axes)/count
        #     residual = apply_along_axes(masked, -self.mean, np.add, axes=axes)
        #     self.stdv = np.sqrt((residual**2).sum(axis=axes)/count)

    def encode(self, data: TensorType, /) -> TensorType:
        broadcast = get_broadcast(data, axis=self.axis)
        return (data - self.mean[broadcast]) / self.stdv[broadcast]

    def decode(self, data: TensorType, /) -> TensorType:
        broadcast = get_broadcast(data, axis=self.axis)
        return data * self.stdv[broadcast] + self.mean[broadcast]


class LinearScaler(BaseEncoder, Generic[TensorType]):
    r"""Maps the interval [x_min, x_max] to [y_min, y_max] (default: [0,1])."""

    # TODO: rewrite as dataclass

    requires_fit: bool = False

    xmin: TensorType  # NDArray[np.number] | Tensor
    xmax: TensorType  # NDArray[np.number] | Tensor
    ymin: TensorType  # NDArray[np.number] | Tensor
    ymax: TensorType  # NDArray[np.number] | Tensor
    scale: TensorType  # NDArray[np.number] | Tensor
    r"""The scaling factor."""
    axis: tuple[int, ...]
    r"""Over which axis to perform the scaling."""

    class Parameters(NamedTuple):
        r"""The parameters of the MinMaxScaler."""

        xmin: TensorLike
        xmax: TensorLike
        ymin: TensorLike
        ymax: TensorLike
        scale: TensorLike
        axis: tuple[int, ...]

        def __repr__(self) -> str:
            r"""Pretty print."""
            return repr_namedtuple(self)

    def __init__(
        self,
        xmin: float | TensorType = 0,
        xmax: float | TensorType = 1,
        *,
        ymin: float | TensorType = 0,
        ymax: float | TensorType = 1,
        axis: Optional[int | tuple[int, ...]] = None,
    ):
        r"""Initialize the MinMaxScaler."""
        super().__init__()
        self.xmin = cast(TensorType, np.array(xmin))
        self.xmax = cast(TensorType, np.array(xmax))
        self.ymin = cast(TensorType, np.array(ymin))
        self.ymax = cast(TensorType, np.array(ymax))
        self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)
        self.axis = (axis,) if isinstance(axis, int) else axis  # type: ignore[assignment]

    def __getitem__(self, item: Any) -> MinMaxScaler:
        r"""Return a slice of the MinMaxScaler."""
        xmin = self.xmin if self.xmin.ndim == 0 else self.xmin[item]
        xmax = self.xmax if self.xmax.ndim == 0 else self.xmax[item]
        ymin = self.ymin if self.ymin.ndim == 0 else self.ymin[item]
        ymax = self.ymax if self.ymax.ndim == 0 else self.ymax[item]

        oldvals = (self.xmin, self.xmax, self.ymin, self.ymax)
        newvals = (xmin, xmax, ymin, ymax)
        assert not all(x.ndim == 0 for x in oldvals)
        lost_ranks = max(x.ndim for x in oldvals) - max(x.ndim for x in newvals)

        encoder = MinMaxScaler(
            ymin, ymax, xmin=xmin, xmax=xmax, axis=self.axis[lost_ranks:]
        )

        encoder._is_fitted = self._is_fitted
        return encoder

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.xmin}, {self.xmax}, {self.ymin}, {self.ymax}, axis={self.axis})"

    @property
    def param(self) -> LinearScaler.Parameters:
        r"""Parameters of the LinearScaler."""
        return self.Parameters(
            xmin=self.xmin,
            xmax=self.xmax,
            ymin=self.ymin,
            ymax=self.ymax,
            scale=self.scale,
            axis=self.axis,
        )

    def fit(self, data: TensorType, /) -> None:
        # TODO: Why does singledispatch not work here? (wrap_func in BaseEncoder)
        # print(type(data), isinstance(data, np.ndarray), isinstance(data, type))
        if isinstance(data, Tensor):
            self.xmin = torch.tensor(self.xmin)
            self.xmax = torch.tensor(self.xmax)
            self.ymin = torch.tensor(self.ymin)
            self.ymax = torch.tensor(self.ymax)
            self.scale = torch.tensor(self.scale)
        else:
            self.xmin = np.array(self.xmin)
            self.xmax = np.array(self.xmax)
            self.ymin = np.array(self.ymin)
            self.ymax = np.array(self.ymax)
            self.scale = np.array(self.scale)

    def encode(self, data: TensorType, /) -> TensorType:
        broadcast = get_broadcast(data, axis=self.axis)

        xmin: TensorType = self.xmin[broadcast] if self.xmin.ndim > 1 else self.xmin
        scale: TensorType = self.scale[broadcast] if self.scale.ndim > 1 else self.scale
        ymin: TensorType = self.ymin[broadcast] if self.ymin.ndim > 1 else self.ymin

        return (data - xmin) * scale + ymin

    def decode(self, data: TensorType, /) -> TensorType:
        broadcast = get_broadcast(data, axis=self.axis)

        xmin = self.xmin[broadcast] if self.xmin.ndim > 1 else self.xmin
        scale = self.scale[broadcast] if self.scale.ndim > 1 else self.scale
        ymin = self.ymin[broadcast] if self.ymin.ndim > 1 else self.ymin

        return (data - ymin) / scale + xmin


class MinMaxScaler(LinearScaler, Generic[TensorType]):
    r"""Maps the interval [x_min, x_max] to [y_min, y_max] (default: [0,1])."""

    requires_fit: bool = True

    ymin: TensorType  # NDArray[np.number] | Tensor
    ymax: TensorType  # NDArray[np.number] | Tensor
    xmin: TensorType  # NDArray[np.number] | Tensor
    xmax: TensorType  # NDArray[np.number] | Tensor
    scale: TensorType  # NDArray[np.number] | Tensor
    r"""The scaling factor."""
    axis: tuple[int, ...]
    r"""Over which axis to perform the scaling."""

    def __init__(
        self,
        ymin: float | TensorType = 0,
        ymax: float | TensorType = 1,
        *,
        xmin: float | TensorType = 0,
        xmax: float | TensorType = 1,
        axis: Optional[int | tuple[int, ...]] = None,
    ):
        r"""Initialize the MinMaxScaler."""
        super().__init__()
        self.xmin = cast(TensorType, np.array(xmin))
        self.xmax = cast(TensorType, np.array(xmax))
        self.ymin = cast(TensorType, np.array(ymin))
        self.ymax = cast(TensorType, np.array(ymax))
        self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)
        self.axis = (axis,) if isinstance(axis, int) else axis  # type: ignore[assignment]

    def __getitem__(self, item: Any) -> MinMaxScaler:
        r"""Return a slice of the MinMaxScaler."""
        xmin = self.xmin if self.xmin.ndim == 0 else self.xmin[item]
        xmax = self.xmax if self.xmax.ndim == 0 else self.xmax[item]
        ymin = self.ymin if self.ymin.ndim == 0 else self.ymin[item]
        ymax = self.ymax if self.ymax.ndim == 0 else self.ymax[item]

        oldvals = (self.xmin, self.xmax, self.ymin, self.ymax)
        newvals = (xmin, xmax, ymin, ymax)
        assert not all(x.ndim == 0 for x in oldvals)
        lost_ranks = max(x.ndim for x in oldvals) - max(x.ndim for x in newvals)

        encoder = MinMaxScaler(
            ymin, ymax, xmin=xmin, xmax=xmax, axis=self.axis[lost_ranks:]
        )

        encoder._is_fitted = self._is_fitted
        return encoder

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(axis={self.axis})"

    @property
    def param(self) -> MinMaxScaler.Parameters:
        r"""Parameters of the MinMaxScaler."""
        return self.Parameters(
            xmin=self.xmin,
            xmax=self.xmax,
            ymin=self.ymin,
            ymax=self.ymax,
            scale=self.scale,
            axis=self.axis,
        )

    def fit(self, data: TensorType, /) -> None:
        # TODO: Why does singledispatch not work here? (wrap_func in BaseEncoder)
        # print(type(data), isinstance(data, np.ndarray), isinstance(data, type))
        if isinstance(data, Tensor):
            self._fit_torch(data)
        else:
            self._fit_numpy(np.asarray(data))

    def _fit_torch(self, data: Tensor, /) -> None:
        r"""Compute the min and max."""
        rank = len(data.shape)
        if self.axis is None:
            self.axis = () if rank == 1 else (-1,)

        selection = [(a % rank) for a in self.axis]
        axes = tuple(k for k in range(rank) if k not in selection)

        mask = torch.isnan(data)
        neginf = torch.tensor(float("-inf"), device=data.device, dtype=data.dtype)
        posinf = torch.tensor(float("+inf"), device=data.device, dtype=data.dtype)

        self.ymin = torch.tensor(self.ymin, device=data.device, dtype=data.dtype)  # type: ignore[assignment]
        self.ymax = torch.tensor(self.ymax, device=data.device, dtype=data.dtype)  # type: ignore[assignment]
        self.xmin = torch.amin(torch.where(mask, posinf, data), dim=axes)  # type: ignore[assignment]
        self.xmax = torch.amax(torch.where(mask, neginf, data), dim=axes)  # type: ignore[assignment]
        self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)

    def _fit_numpy(self, data: np.ndarray, /) -> None:
        r"""Compute the min and max."""
        rank = len(data.shape)
        if self.axis is None:
            self.axis = () if rank == 1 else (-1,)

        selection = [(a % rank) for a in self.axis]
        axes = tuple(k for k in range(rank) if k not in selection)
        # axes = self.axis
        self.ymin = cast(TensorType, np.array(self.ymin, dtype=data.dtype))
        self.ymax = cast(TensorType, np.array(self.ymax, dtype=data.dtype))
        self.xmin = cast(TensorType, np.nanmin(data, axis=axes))
        self.xmax = cast(TensorType, np.nanmax(data, axis=axes))
        self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)

    def encode(self, data: TensorType, /) -> TensorType:
        broadcast = get_broadcast(data, axis=self.axis)

        xmin: TensorType = self.xmin[broadcast] if self.xmin.ndim > 1 else self.xmin
        scale: TensorType = self.scale[broadcast] if self.scale.ndim > 1 else self.scale
        ymin: TensorType = self.ymin[broadcast] if self.ymin.ndim > 1 else self.ymin

        return (data - xmin) * scale + ymin

    def decode(self, data: TensorType, /) -> TensorType:
        broadcast = get_broadcast(data, axis=self.axis)

        xmin = self.xmin[broadcast] if self.xmin.ndim > 1 else self.xmin
        scale = self.scale[broadcast] if self.scale.ndim > 1 else self.scale
        ymin = self.ymin[broadcast] if self.ymin.ndim > 1 else self.ymin

        return (data - ymin) / scale + xmin


class LogEncoder(BaseEncoder):
    r"""Encode data on logarithmic scale.

    Uses base 2 by default for lower numerical error and fast computation.
    """

    threshold: NDArray
    replacement: NDArray

    def fit(self, data: NDArray, /) -> None:
        assert np.all(data >= 0)

        mask = data == 0
        self.threshold = data[~mask].min()
        self.replacement = np.log2(self.threshold / 2)

    def encode(self, data: NDArray, /) -> NDArray:
        result = data.copy()
        mask = data <= 0
        result[:] = np.where(mask, self.replacement, np.log2(data))
        return result

    def decode(self, data: NDArray, /) -> NDArray:
        result = 2**data
        mask = result < self.threshold
        result[:] = np.where(mask, 0, result)
        return result


class LogitEncoder(BaseEncoder):
    """Logit encoder."""

    requires_fit = False

    def encode(self, data: DataFrame, /) -> DataFrame:
        assert all((data > 0) & (data < 1))
        return np.log(data / (1 - data))

    def decode(self, data: DataFrame, /) -> DataFrame:
        return np.clip(1 / (1 + np.exp(-data)), 0, 1)


class FloatEncoder(BaseEncoder):
    r"""Converts all columns of DataFrame to float32."""

    dtypes: Series = None
    r"""The original dtypes."""

    def __init__(self, dtype: str = "float32") -> None:
        self.target_dtype = dtype
        super().__init__()

    def fit(self, data: PandasObject, /) -> None:
        if isinstance(data, DataFrame):
            self.dtypes = data.dtypes
        elif isinstance(data, (Series, Index)):
            self.dtypes = data.dtype
        # elif hasattr(data, "dtype"):
        #     self.dtypes = data.dtype
        # elif hasattr(data, "dtypes"):
        #     self.dtypes = data.dtype
        else:
            raise TypeError(f"Cannot get dtype of {type(data)}")

    def encode(self, data: PandasObject, /) -> PandasObject:
        return data.astype(self.target_dtype)

    def decode(self, data: PandasObject, /) -> PandasObject:
        return data.astype(self.dtypes)


class IntEncoder(BaseEncoder):
    r"""Converts all columns of DataFrame to int32."""

    dtypes: Series = None
    r"""The original dtypes."""

    def fit(self, data: PandasObject, /) -> None:
        self.dtypes = data.dtypes

    def encode(self, data: PandasObject, /) -> PandasObject:
        return data.astype("int32")

    def decode(self, data, /):
        return data.astype(self.dtypes)


class TensorSplitter(BaseEncoder):
    r"""Split tensor along specified axis."""

    lengths: list[int]
    numdims: list[int]
    axis: int
    maxdim: int
    indices_or_sections: int | list[int]

    def __init__(
        self, *, indices_or_sections: int | list[int] = 1, axis: int = 0
    ) -> None:
        r"""Concatenate tensors along the specified axis."""
        super().__init__()
        self.axis = axis
        self.indices_or_sections = indices_or_sections

    @overload
    def encode(self, data: Tensor, /) -> list[Tensor]:
        ...

    @overload
    def encode(self, data: NDArray, /) -> list[NDArray]:
        ...

    def encode(self, data, /):
        if isinstance(data, Tensor):
            return torch.tensor_split(data, self.indices_or_sections, dim=self.axis)
        return np.array_split(data, self.indices_or_sections, dim=self.axis)  # type: ignore[call-overload]

    @overload
    def decode(self, data: list[Tensor], /) -> Tensor:
        ...

    @overload
    def decode(self, data: list[NDArray], /) -> NDArray:
        ...

    def decode(self, data, /):
        if isinstance(data[0], Tensor):
            return torch.cat(data, dim=self.axis)
        return np.concatenate(data, axis=self.axis)


class TensorConcatenator(BaseEncoder):
    r"""Concatenate multiple tensors.

    Useful for concatenating encoders for multiple inputs.
    """

    lengths: list[int]
    numdims: list[int]
    axis: int
    maxdim: int

    def __init__(self, axis: int = 0) -> None:
        r"""Concatenate tensors along the specified axis."""
        super().__init__()
        self.axis = axis

    def fit(self, data: tuple[Tensor, ...], /) -> None:
        self.numdims = [d.ndim for d in data]
        self.maxdim = max(self.numdims)
        # pad dimensions if necessary
        arrays = [d[(...,) + (None,) * (self.maxdim - d.ndim)] for d in data]
        # store the lengths of the slices
        self.lengths = [x.shape[self.axis] for x in arrays]

    def encode(self, data: tuple[Tensor, ...], /) -> Tensor:
        return torch.cat(
            [d[(...,) + (None,) * (self.maxdim - d.ndim)] for d in data], dim=self.axis
        )

    def decode(self, data: Tensor, /) -> tuple[Tensor, ...]:
        result = torch.split(data, self.lengths, dim=self.axis)
        return tuple(x.squeeze() for x in result)
