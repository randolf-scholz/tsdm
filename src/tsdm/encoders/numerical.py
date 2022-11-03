r"""Numerical Transformations, Standardization, Log-Transforms, etc."""

from __future__ import annotations

__all__ = [
    # Classes
    "BoundaryEncoder",
    "BoxCoxEncoder",
    "LogEncoder",
    "LogitEncoder",
    "LogitBoxCoxEncoder",
    "MinMaxScaler",
    "Standardizer",
    "TensorConcatenator",
    "TensorSplitter",
]

from collections.abc import Callable
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
import torch
from numpy import pi as PI
from numpy.typing import NDArray
from pandas import NA, DataFrame, Index, Series
from scipy.optimize import minimize
from scipy.special import erfinv
from torch import Tensor

from tsdm.encoders.base import BaseEncoder
from tsdm.utils.strings import repr_namedtuple
from tsdm.utils.types import PandasObject

TensorLike: TypeAlias = Tensor | NDArray | DataFrame | Series
r"""Type Hint for tensor-like objects."""
TensorType = TypeVar("TensorType", Tensor, np.ndarray, DataFrame, Series)
r"""TypeVar for tensor-like objects."""


def get_broadcast(
    data: Any, /, *, axis: tuple[int, ...] | None
) -> tuple[slice | None, ...]:
    r"""Create an indexer for axis specific broadcasting.

    Example
    -------
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


class Standardizer(BaseEncoder, Generic[TensorType]):
    r"""A StandardScalar that works with batch dims."""

    mean: Any
    r"""The mean value."""
    stdv: Any
    r"""The standard-deviation."""
    ignore_nan: bool = True
    r"""Whether to ignore nan-values while fitting."""
    axis: tuple[int, ...] | None
    r"""The axis to perform the scaling. If None, automatically select the axis."""

    class Parameters(NamedTuple):
        r"""The parameters of the StandardScalar."""

        mean: TensorLike
        stdv: TensorLike
        axis: tuple[int, ...] | None

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
        axis: Optional[int | tuple[int, ...]] = None,
    ):
        super().__init__()
        self.ignore_nan = ignore_nan
        self.axis = (axis,) if isinstance(axis, int) else axis
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
        if self.mean is None:
            raise RuntimeError("Needs to be fitted first!")

        self.LOGGER.debug("Encoding data %s", data)
        broadcast = get_broadcast(data, axis=self.axis)
        self.LOGGER.debug("Broadcasting to %s", broadcast)

        return (data - self.mean[broadcast]) / self.stdv[broadcast]

    def decode(self, data: TensorType, /) -> TensorType:
        if self.mean is None:
            raise RuntimeError("Needs to be fitted first!")

        self.LOGGER.debug("Encoding data %s", data)
        broadcast = get_broadcast(data, axis=self.axis)
        self.LOGGER.debug("Broadcasting to %s", broadcast)

        return data * self.stdv[broadcast] + self.mean[broadcast]


class MinMaxScaler(BaseEncoder, Generic[TensorType]):
    r"""A MinMaxScaler that works with batch dims and both numpy/torch."""

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
        /,
        ymin: Optional[float | TensorType] = None,
        ymax: Optional[float | TensorType] = None,
        xmin: Optional[float | TensorType] = None,
        xmax: Optional[float | TensorType] = None,
        *,
        axis: Optional[int | tuple[int, ...]] = None,
    ):
        r"""Initialize the MinMaxScaler."""
        super().__init__()
        self.xmin = cast(TensorType, np.array(0.0 if xmin is None else xmin))
        self.xmax = cast(TensorType, np.array(1.0 if xmax is None else xmax))
        self.ymin = cast(TensorType, np.array(0.0 if ymin is None else ymin))
        self.ymax = cast(TensorType, np.array(1.0 if ymax is None else ymax))
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
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, axis=self.axis[lost_ranks:]
        )

        encoder._is_fitted = self._is_fitted
        return encoder

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(axis={self.axis})"

    @property
    def param(self) -> Parameters:
        r"""Parameters of the MinMaxScaler."""
        return self.Parameters(
            self.xmin, self.xmax, self.ymin, self.ymax, self.scale, self.axis
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
        self.LOGGER.debug("Encoding data %s", data)
        broadcast = get_broadcast(data, axis=self.axis)
        self.LOGGER.debug("Broadcasting to %s", broadcast)

        xmin: TensorType = self.xmin[broadcast] if self.xmin.ndim > 1 else self.xmin
        scale: TensorType = self.scale[broadcast] if self.scale.ndim > 1 else self.scale
        ymin: TensorType = self.ymin[broadcast] if self.ymin.ndim > 1 else self.ymin

        return (data - xmin) * scale + ymin

    def decode(self, data: TensorType, /) -> TensorType:
        self.LOGGER.debug("Decoding data %s", data)
        broadcast = get_broadcast(data, axis=self.axis)
        self.LOGGER.debug("Broadcasting to %s", broadcast)

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


class FloatEncoder(BaseEncoder):
    r"""Converts all columns of DataFrame to float32."""

    dtypes: Series = None
    r"""The original dtypes."""

    def __init__(self, dtype: str = "float32"):
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


class BoundaryEncoder(BaseEncoder):
    r"""Clip or mask values outside a given range.

    Parameters
    ----------
    lower : lower bound - scalar or array-like
    upper : upper bound - scalar or array-like
    mode: str
        'clip': clip to boundary
        'mask': mask values outside of boundary with nan
    axis: int | tuple[..., int]
    """

    lower: float | np.ndarray
    upper: float | np.ndarray
    axis: int | tuple[int, ...] = -1
    mode: Literal["mask", "clip"] = "mask"
    _nan: float = np.nan

    def __init__(self, lower, upper, mode="mask", axis=-1):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.axis = axis
        self.mode = mode

    def fit(self, data):
        # TODO: make _nan adapt to real data type!
        if isinstance(data, Series | DataFrame):
            self._nan = NA
        else:
            self._nan = float("nan")

    def encode(self, data: DataFrame) -> DataFrame:
        if self.mode == "mask":
            data = data.where(data < self.lower, self._nan)
            data = data.where(data > self.upper, self._nan)
            return data
        if self.mode == "clip":
            data = data.where(data < self.lower, self.lower)
            data = data.where(data > self.upper, self.upper)
            return data

        raise ValueError(f"Unknown mode {self.mode}")

    def decode(self, data: DataFrame) -> DataFrame:
        return data


class BoxCoxEncoder(BaseEncoder):
    r"""Encode data on logarithmic scale with offset.

    .. math:: x ↦ \log(x+c)

    We consider multiple ideas for how to fit the parameter $c$

    1. Half the minimal non-zero value: `c = min(data[data>0])/2`
    2. Square of the first quartile divided by the third quartile (Stahle 2002)
    3. Value which minimizes the Wasserstein distance to
        - a mean-0, variance-1 uniform distribution
        - a mean-0, variance-1 normal distribution
    """

    METHOD: TypeAlias = Literal["minimum", "quartile", "match-normal", "match-uniform"]
    AVAILABLE_METHODS = [None, "minimum", "quartile", "match-normal", "match-uniform"]

    method: Optional[METHOD] = "match-uniform"
    offset: np.ndarray

    def __init__(
        self,
        *,
        method: Optional[METHOD] = "match-uniform",
        initial_param: Optional[np.ndarray] = None,
    ) -> None:

        if method not in self.AVAILABLE_METHODS:
            raise ValueError(f"{method=} unknown. Available: {self.AVAILABLE_METHODS}")
        if method is None and initial_param is None:
            raise ValueError("Needs to provide initial param if no fitting.")

        self.method = method
        self.initial_param = initial_param
        super().__init__()

    @staticmethod
    def construct_loss_wasserstein_uniform(
        x: NDArray, a: float = -np.sqrt(3), b: float = +np.sqrt(3)  # noqa: B008
    ) -> Callable[[NDArray], NDArray]:
        r"""Construct the loss for the Uniform distribution.

        .. math::
            W₂² = ∑ₖ [αₖxₖ² -2βₖxₖ + αₖC] = ∑ₖ αₖ[xₖ² -2(βₖ/αₖ)xₖ + C]
            F^{-1}(q) &= a + (b-a)q
            β &= ∫ F^{-1}(q)dq = aq + ½(b-a)q²
            C &= ∫_0^1 F^{-1}(q)^2 dq = ⅓(a^2 + ab + b^2)

        Also note: (1, 1; -1, 1)(a,b) = (2, 0; 0, √12) (μ, σ)
        Hence: a = μ-√3σ, b = μ+√3σ
        And: μ = ½(a+b), σ² = (a-b)²/12

        """
        if (a, b) == (-np.sqrt(3), +np.sqrt(3)):
            C = 1.0

            def integrate_quantile(q):
                return np.sqrt(3) * q * (q - 1)

        else:
            C = (a**2 + a * b + b**2) / 3

            def integrate_quantile(q):
                return a * q + (b - a) * q**2 / 2

        mask = np.isnan(x)
        unique, counts = np.unique(x[~mask], return_counts=True)
        α = counts / np.sum(counts)
        p = np.insert(np.cumsum(α), 0, 0).clip(0, 1)
        β = integrate_quantile(p[1:]) - integrate_quantile(p[:-1])
        μ = (b + a) / 2
        σ = abs(b - a) / np.sqrt(12)

        def fun(c: NDArray) -> NDArray:
            u = np.log(np.add.outer(c, unique))
            # transform to target loc-scale
            mean = np.mean(u, axis=-1, keepdims=True)
            stdv = np.std(u, axis=-1, keepdims=True)
            y = (u - mean + μ) * (σ / stdv)
            return np.einsum("...i, i -> ...", y**2 - 2 * (β / α) * y + C, α)

        return fun

    @staticmethod
    def construct_loss_wasserstein_normal(
        x: NDArray, μ: float = 0.0, σ: float = 1.0
    ) -> Callable[[NDArray], NDArray]:
        r"""Construct the loss for the Normal distribution.

        .. math::
            W₂² = ∑ₖ [αₖxₖ² -2βₖxₖ + αₖC] = ∑ₖ αₖ[xₖ² -2(βₖ/αₖ)xₖ + C]
            F^{-1}(q) &= μ + σ√2\erf^{-1}(2q-1)
            β &= ∫_a^b F^{-1}(q)dq = (b-a)μ - σ/√(2PI) (e^{-\erf^{-1}(2b-1)^2} - e^{-\erf^{-1}(2a-1)^2}
            C &= ∫_0^1 F^{-1}(q)^2 dq = μ^2 + σ^2
        """
        if (μ, σ) == (0, 1):
            C = 1.0

            def integrate_quantile(q):
                return -np.exp(-erfinv(2 * q - 1) ** 2) / np.sqrt(2 * PI)

        else:
            C = μ**2 + σ**2

            def integrate_quantile(q):
                return μ * q - σ * np.exp(-erfinv(2 * q - 1) ** 2) / np.sqrt(2 * PI)

        mask = np.isnan(x)
        unique, counts = np.unique(x[~mask], return_counts=True)
        α = counts / np.sum(counts)
        p = np.insert(np.cumsum(α), 0, 0).clip(0, 1)
        β = integrate_quantile(p[1:]) - integrate_quantile(p[:-1])

        def fun(c: NDArray) -> NDArray:
            u = np.log(np.add.outer(c, unique))
            # transform to target loc-scale
            mean = np.mean(u, axis=-1, keepdims=True)
            stdv = np.std(u, axis=-1, keepdims=True)
            y = (u - mean + μ) * (σ / stdv)
            return np.einsum("...i, i -> ...", y**2 - 2 * (β / α) * y + C, α)

        return fun

    def fit(self, data: DataFrame, /) -> None:
        assert all(np.isnan(data) | (data >= 0)), f"{data=}"

        match self.method:
            case None:
                assert self.initial_param is not None
                self.offset = self.initial_param
            case "minimum":
                self.offset = data[data > 0].min() / 2
            case "quartile":
                self.offset = (np.quantile(data, 0.25) / np.quantile(data, 0.75)) ** 2
            case "match-uniform":
                fun = self.construct_loss_wasserstein_uniform(data)
                x0 = np.array([1.0])
                sol = minimize(
                    fun,
                    x0,
                    method="trust-constr",
                    # jac=jac,
                    # hess=hess,
                    bounds=[(0, np.inf)],
                    options={"disp": True},
                )
                self.offset = sol.x.squeeze()
            case "match-normal":
                fun = self.construct_loss_wasserstein_normal(data)
                x0 = np.array([1.0])
                sol = minimize(
                    fun,
                    x0,
                    method="trust-constr",
                    # jac=jac,
                    # hess=hess,
                    bounds=[(0, np.inf)],
                    options={"disp": True},
                )
                self.offset = sol.x.squeeze()
            case _:
                raise ValueError(f"Unknown method {self.method}")

    def encode(self, data: DataFrame, /) -> DataFrame:
        assert all(np.isnan(data) | (data >= 0)), f"{data=}"
        return np.log(data + self.offset)

    def decode(self, data: DataFrame, /) -> DataFrame:
        return np.maximum(np.exp(data) - self.offset, 0)


class LogitEncoder(BaseEncoder):
    """Logit encoder."""

    requires_fit = False

    def encode(self, data: DataFrame, /) -> DataFrame:
        assert all((data > 0) & (data < 1))
        return np.log(data / (1 - data))

    def decode(self, data: DataFrame, /) -> DataFrame:
        return np.clip(1 / (1 + np.exp(-data)), 0, 1)


class LogitBoxCoxEncoder(BaseEncoder):
    r"""Encode data on logarithmic scale with offset.

    .. math:: x ↦ \log(x+c)

    We consider multiple ideas for how to fit the parameter $c$

    1. Half the minimal non-zero value: `c = min(data[data>0])/2`
    2. Square of the first quartile divided by the third quartile (Stahle 2002)
    3. Value which minimizes the Wasserstein distance to
        - a mean-0, variance-1 uniform distribution
        - a mean-0, variance-1 normal distribution
    """

    METHOD: TypeAlias = Literal["minimum", "quartile", "match-normal", "match-uniform"]
    AVAILABLE_METHODS = [None, "minimum", "quartile", "match-normal", "match-uniform"]

    method: Optional[METHOD] = "match-normal"
    offset: np.ndarray

    def __init__(
        self,
        *,
        method: Optional[METHOD] = "match-normal",
        initial_param: Optional[np.ndarray] = None,
    ) -> None:

        if method not in self.AVAILABLE_METHODS:
            raise ValueError(f"{method=} unknown. Available: {self.AVAILABLE_METHODS}")
        if method is None and initial_param is None:
            raise ValueError("Needs to provide initial param if no fitting.")

        self.method = method
        self.initial_param = initial_param
        super().__init__()

    @staticmethod
    def construct_loss_wasserstein_uniform(
        x: NDArray, a: float = -np.sqrt(3), b: float = +np.sqrt(3)  # noqa: B008
    ) -> Callable[[NDArray], NDArray]:
        r"""Construct the loss for the Uniform distribution.

        .. math::
            W₂² = ∑ₖ [αₖxₖ² -2βₖxₖ + αₖC] = ∑ₖ αₖ[xₖ² -2(βₖ/αₖ)xₖ + C]
            F^{-1}(q) &= a + (b-a)q
            β &= ∫ F^{-1}(q)dq = aq + ½(b-a)q²
            C &= ∫_0^1 F^{-1}(q)^2 dq = ⅓(a^2 + ab + b^2)

        Also note: (1, 1; -1, 1)(a,b) = (2, 0; 0, √12) (μ, σ)
        Hence: a = μ-√3σ, b = μ+√3σ
        And: μ = ½(a+b), σ² = (a-b)²/12

        """
        if (a, b) == (-np.sqrt(3), +np.sqrt(3)):
            C = 1.0

            def integrate_quantile(q):
                return np.sqrt(3) * q * (q - 1)

        else:
            C = (a**2 + a * b + b**2) / 3

            def integrate_quantile(q):
                return a * q + (b - a) * q**2 / 2

        mask = np.isnan(x)
        unique, counts = np.unique(x[~mask], return_counts=True)
        α = counts / np.sum(counts)
        p = np.insert(np.cumsum(α), 0, 0).clip(0, 1)
        β = integrate_quantile(p[1:]) - integrate_quantile(p[:-1])
        μ = (b + a) / 2
        σ = abs(b - a) / np.sqrt(12)

        def fun(c: NDArray) -> NDArray:
            u = np.log(np.add.outer(c, unique) / (1 + np.add.outer(c, -unique)))
            # transform to target loc-scale
            mean = np.mean(u, axis=-1, keepdims=True)
            stdv = np.std(u, axis=-1, keepdims=True)
            y = (u - mean + μ) * (σ / stdv)
            return np.einsum("...i, i -> ...", y**2 - 2 * (β / α) * y + C, α)

        return fun

    @staticmethod
    def construct_loss_wasserstein_normal(
        x: NDArray, μ: float = 0.0, σ: float = 1.0
    ) -> Callable[[NDArray], NDArray]:
        r"""Construct the loss for the Normal distribution.

        .. math::
            W₂² = ∑ₖ [αₖxₖ² -2βₖxₖ + αₖC] = ∑ₖ αₖ[xₖ² -2(βₖ/αₖ)xₖ + C]
            F^{-1}(q) &= μ + σ√2\erf^{-1}(2q-1)
            β &= ∫_a^b F^{-1}(q)dq = (b-a)μ - σ/√(2PI) (e^{-\erf^{-1}(2b-1)^2} - e^{-\erf^{-1}(2a-1)^2}
            C &= ∫_0^1 F^{-1}(q)^2 dq = μ^2 + σ^2
        """
        if (μ, σ) == (0, 1):
            C = 1.0

            def integrate_quantile(q):
                return -np.exp(-erfinv(2 * q - 1) ** 2) / np.sqrt(2 * PI)

        else:
            C = μ**2 + σ**2

            def integrate_quantile(q):
                return μ * q - σ * np.exp(-erfinv(2 * q - 1) ** 2) / np.sqrt(2 * PI)

        mask = np.isnan(x)
        unique, counts = np.unique(x[~mask], return_counts=True)
        α = counts / np.sum(counts)
        p = np.insert(np.cumsum(α), 0, 0).clip(0, 1)
        β = integrate_quantile(p[1:]) - integrate_quantile(p[:-1])

        def fun(c: NDArray) -> NDArray:
            u = np.log(np.add.outer(c, unique) / (1 + np.add.outer(c, -unique)))
            # transform to target loc-scale
            mean = np.mean(u, axis=-1, keepdims=True)
            stdv = np.std(u, axis=-1, keepdims=True)
            y = (u - mean + μ) * (σ / stdv)
            return np.einsum("...i, i -> ...", y**2 - 2 * (β / α) * y + C, α)

        return fun

    def fit(self, data: DataFrame, /) -> None:
        assert all(np.isnan(data) | ((data >= 0) & (data <= 1))), f"{data=}"

        match self.method:
            case None:
                assert self.initial_param is not None
                self.offset = self.initial_param
            case "minimum":
                self.offset = data[data > 0].min() / 2
            case "quartile":
                self.offset = (np.quantile(data, 0.25) / np.quantile(data, 0.75)) ** 2
            case "match-uniform":
                fun = self.construct_loss_wasserstein_uniform(data)
                x0 = np.array([1.0])
                sol = minimize(
                    fun,
                    x0,
                    method="trust-constr",
                    # jac=jac,
                    # hess=hess,
                    bounds=[(0, np.inf)],
                    options={"disp": True},
                )
                self.offset = sol.x.squeeze()
            case "match-normal":
                fun = self.construct_loss_wasserstein_normal(data)
                x0 = np.array([1.0])
                sol = minimize(
                    fun,
                    x0,
                    method="trust-constr",
                    # jac=jac,
                    # hess=hess,
                    bounds=[(0, np.inf)],
                    options={"disp": True},
                )
                self.offset = sol.x.squeeze()
            case _:
                raise ValueError(f"Unknown method {self.method}")

    def encode(self, data: DataFrame, /) -> DataFrame:
        assert all(np.isnan(data) | ((data >= 0) & (data <= 1))), f"{data=}"
        return np.log((data + self.offset) / (1 - data + self.offset))

    def decode(self, data: DataFrame, /) -> DataFrame:
        ey = np.exp(-data)
        r = (1 + (1 - ey) * self.offset) / (1 + ey)
        return np.clip(r, 0, 1)
