r"""#TODO add module summary line.

#TODO add module description.
"""
from __future__ import annotations

__all__ = [
    # Classes
    "Standardizer",
    "MinMaxScaler",
    "LogEncoder",
]

import logging
from functools import singledispatchmethod
from typing import Any, Generic, NamedTuple, Optional, TypeVar, Union

import numpy as np
import torch
from numpy.typing import NDArray
from pandas import DataFrame, Index, Series
from torch import Tensor

from tsdm.encoders.modular.generic import BaseEncoder
from tsdm.util.strings import repr_namedtuple

__logger__ = logging.getLogger(__name__)


PandasObject = Union[Index, Series, DataFrame]
r"""Type Hint for pandas objects."""

TensorLike = Union[Tensor, NDArray, DataFrame, Series]
r"""Type Hint for tensor-like objects."""
TensorType = TypeVar("TensorType", Tensor, NDArray, DataFrame, Series)
r"""TypeVar for tensor-like objects."""


def get_broadcast(data: Any, axis: tuple[int, ...]) -> tuple[Union[None, slice], ...]:
    r"""Get the broadcast transform.

    Example
    -------
    data is (2,3,4,5,6,7)
    axis is (0,2,-1)
    broadcast is (:, None, :, None, None, :)
    then, given a tensor x of shape (2, 4, 7), we can perform
    element-wise operations via data + x[broadcast]
    """
    rank = len(data.shape)
    axes = list(range(rank))
    axis = tuple(a % rank for a in axis)
    broadcast = tuple(slice(None) if a in axis else None for a in axes)
    return broadcast


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
        axis: tuple[int, ...]

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
        axis: Optional[Union[int, tuple[int, ...]]] = None,
    ):
        super().__init__()
        self.ignore_nan = ignore_nan
        self.axis = (axis,) if isinstance(axis, int) else axis  # type: ignore
        self.mean = mean
        self.stdv = stdv

    def __repr__(self) -> str:
        r"""Pretty print."""
        return f"{self.__class__.__name__}(axis={self.axis})"

    def __getitem__(self, item: Any) -> Standardizer:
        r"""Return a slice of the Standardizer."""
        encoder = Standardizer(
            mean=self.mean[item], stdv=self.stdv[item], axis=self.axis[1:]
        )
        encoder._is_fitted = self._is_fitted
        return encoder

    @property
    def param(self) -> Parameters:
        r"""Parameters of the Standardizer."""
        return self.Parameters(self.mean, self.stdv, self.axis)

    def fit(self, data: TensorType, /) -> None:
        r"""Compute the mean and stdv."""
        rank = len(data.shape)
        if self.axis is None:
            self.axis = () if rank == 1 else (-1,)

        selection = [(a % rank) for a in self.axis]
        axes = tuple(k for k in range(rank) if k not in selection)

        if isinstance(data, Tensor):
            if self.ignore_nan:
                self.mean = torch.nanmean(data, dim=axes)
                self.stdv = torch.sqrt(torch.nanmean((data - self.mean) ** 2, dim=axes))
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
        r"""Encode the input."""
        if self.mean is None:
            raise RuntimeError("Needs to be fitted first!")

        __logger__.debug("Encoding data %s", data)
        broadcast = get_broadcast(data, self.axis)
        __logger__.debug("Broadcasting to %s", broadcast)

        return (data - self.mean[broadcast]) / self.stdv[broadcast]

    def decode(self, data: TensorType, /) -> TensorType:
        r"""Decode the input."""
        if self.mean is None:
            raise RuntimeError("Needs to be fitted first!")

        __logger__.debug("Encoding data %s", data)
        broadcast = get_broadcast(data, self.axis)
        __logger__.debug("Broadcasting to %s", broadcast)

        return data * self.stdv[broadcast] + self.mean[broadcast]


class MinMaxScaler(BaseEncoder, Generic[TensorType]):
    r"""A MinMaxScaler that works with batch dims and both numpy/torch."""

    xmin: Union[NDArray, Tensor]
    xmax: Union[NDArray, Tensor]
    ymin: Union[NDArray, Tensor]
    ymax: Union[NDArray, Tensor]
    scale: Union[NDArray, Tensor]
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
        /,
        ymin: Optional[Union[float, TensorType]] = None,
        ymax: Optional[Union[float, TensorType]] = None,
        xmin: Optional[Union[float, TensorType]] = None,
        xmax: Optional[Union[float, TensorType]] = None,
        *,
        axis: Optional[Union[int, tuple[int, ...]]] = None,
    ):
        r"""Initialize the MinMaxScaler.

        Parameters
        ----------
        ymin
        ymax
        axis
        """
        super().__init__()
        self.xmin = np.array(0.0 if xmin is None else xmin)
        self.xmax = np.array(1.0 if xmax is None else xmax)
        self.ymin = np.array(0.0 if ymin is None else ymin)
        self.ymax = np.array(1.0 if ymax is None else ymax)
        self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)
        self.axis = (axis,) if isinstance(axis, int) else axis  # type: ignore

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
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, axis=self.axis[lost_ranks:]
        )

        encoder._is_fitted = self._is_fitted
        return encoder

    def __repr__(self) -> str:
        r"""Pretty print."""
        return f"{self.__class__.__name__}(axis={self.axis})"

    @property
    def param(self) -> Parameters:
        r"""Parameters of the MinMaxScaler."""
        return self.Parameters(
            self.xmin, self.xmax, self.ymin, self.ymax, self.scale, self.axis
        )

    @singledispatchmethod
    def fit(self, data: TensorType, /) -> None:  # type: ignore[override]
        r"""Compute the min and max."""
        array = np.asarray(data)
        rank = len(array.shape)
        if self.axis is None:
            self.axis = () if rank == 1 else (-1,)

        selection = [(a % rank) for a in self.axis]
        axes = tuple(k for k in range(rank) if k not in selection)
        # axes = self.axis
        self.ymin = np.array(self.ymin, dtype=array.dtype)
        self.ymax = np.array(self.ymax, dtype=array.dtype)
        self.xmin = np.nanmin(array, axis=axes)
        self.xmax = np.nanmax(array, axis=axes)
        self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)

    @fit.register(torch.Tensor)
    def _(self, data: Tensor, /) -> None:
        r"""Compute the min and max."""
        mask = torch.isnan(data)
        self.ymin = torch.tensor(self.ymin, device=data.device, dtype=data.dtype)
        self.ymax = torch.tensor(self.ymax, device=data.device, dtype=data.dtype)
        neginf = torch.tensor(float("-inf"), device=data.device, dtype=data.dtype)
        posinf = torch.tensor(float("+inf"), device=data.device, dtype=data.dtype)

        self.xmin = torch.amin(torch.where(mask, posinf, data), dim=self.axis)
        self.xmax = torch.amax(torch.where(mask, neginf, data), dim=self.axis)
        self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)

    def encode(self, data: TensorType, /) -> TensorType:
        r"""Encode the input."""
        __logger__.debug("Encoding data %s", data)
        broadcast = get_broadcast(data, self.axis)
        __logger__.debug("Broadcasting to %s", broadcast)

        xmin = self.xmin[broadcast] if self.xmin.ndim > 1 else self.xmin
        scale = self.scale[broadcast] if self.scale.ndim > 1 else self.scale
        ymin = self.ymin[broadcast] if self.ymin.ndim > 1 else self.ymin

        return (data - xmin) * scale + ymin

    def decode(self, data: TensorType, /) -> TensorType:
        r"""Decode the input."""
        __logger__.debug("Decoding data %s", data)
        broadcast = get_broadcast(data, self.axis)
        __logger__.debug("Broadcasting to %s", broadcast)

        xmin = self.xmin[broadcast] if self.xmin.ndim > 1 else self.xmin
        scale = self.scale[broadcast] if self.scale.ndim > 1 else self.scale
        ymin = self.ymin[broadcast] if self.ymin.ndim > 1 else self.ymin

        return (data - ymin) / scale + xmin

    # @singledispatchmethod
    # def fit(self, data: Union[Tensor, np.ndarray], /) -> None:
    #     r"""Compute the min and max."""
    #     return self.fit(np.asarray(data))
    #
    # @fit.register(Tensor)
    # def _(self, data: Tensor) -> Tensor:
    #     r"""Compute the min and max."""
    #     mask = torch.isnan(data)
    #     self.xmin = torch.min(
    #         torch.where(mask, torch.tensor(float("+inf")), data), dim=self.axis
    #     )
    #     self.xmax = torch.max(
    #         torch.where(mask, torch.tensor(float("-inf")), data), dim=self.axis
    #     )
    #     self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)

    # @encode.register(Tensor)  # type: ignore[no-redef]
    # def _(self, data: Tensor) -> Tensor:
    #     r"""Encode the input."""
    #     return self.scale * (data - self.xmin) + self.ymin
    #
    # @encode.register(Tensor)  # type: ignore[no-redef]
    # def _(self, data: NDArray) -> NDArray:
    #     r"""Encode the input."""
    #     return self.scale * (data - self.xmin) + self.ymin

    # @singledispatchmethod

    # @decode.register(Tensor)  # type: ignore[no-redef]
    # def _(self, data, /):
    #     r"""Decode the input."""
    #     return (1 / self.scale) * (data - self.ymin) + self.xmin
    #
    # @decode.register(Tensor)  # type: ignore[no-redef]
    # def _(self, data, /):
    #     r"""Decode the input."""
    #     return (1 / self.scale) * (data - self.ymin) + self.xmin

    # def fit(self, data, /) -> None:
    #     r"""Compute the min and max."""
    #     data = np.asarray(data)
    #     self.xmax = np.nanmax(data, axis=self.axis)
    #     self.xmin = np.nanmin(data, axis=self.axis)
    #     print(self.xmax, self.xmin)
    #     self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)
    #
    # def encode(self, data, /):
    #     r"""Encode the input."""
    #     return self.scale * (data - self.xmin) + self.ymin
    #
    # def decode(self, data, /):
    #     r"""Decode the input."""
    #     return (1 / self.scale) * (data - self.ymin) + self.xmin


class LogEncoder(BaseEncoder):
    r"""Encode data on logarithmic scale.

    Uses base 2 by default for lower numerical error and fast computation.
    """

    threshold: NDArray
    replacement: NDArray

    def fit(self, data: NDArray, /) -> None:
        r"""Fit the encoder to the data."""
        assert np.all(data >= 0)

        mask = data == 0
        self.threshold = data[~mask].min()
        self.replacement = np.log2(self.threshold / 2)

    def encode(self, data: NDArray, /) -> NDArray:
        r"""Encode data on logarithmic scale."""
        result = data.copy()
        mask = data <= 0
        result[:] = np.where(mask, self.replacement, np.log2(data))
        return result

    def decode(self, data: NDArray, /) -> NDArray:
        r"""Decode data on logarithmic scale."""
        result = 2**data
        mask = result < self.threshold
        result[:] = np.where(mask, 0, result)
        return result