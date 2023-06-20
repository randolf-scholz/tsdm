r"""Numerical Transformations, Standardization, Log-Transforms, etc."""

from __future__ import annotations

__all__ = [
    # functions
    "get_broadcast",
    "get_reduced_axes",
    # Classes
    "BoundaryEncoder",
    "LinearScaler",
    "LogEncoder",
    "LogitEncoder",
    "MinMaxScaler",
    "StandardScaler",
    "TensorConcatenator",
    "TensorSplitter",
]

import operator
from dataclasses import KW_ONLY, dataclass, field
from types import EllipsisType, ModuleType
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    NamedTuple,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pandas import DataFrame, Index, Series
from torch import Tensor
from typing_extensions import Self

from tsdm.encoders.base import BaseEncoder
from tsdm.types.aliases import PandasObject
from tsdm.utils.strings import repr_dataclass, repr_namedtuple

TensorLike: TypeAlias = Tensor | NDArray | DataFrame | Series
r"""Type Hint for tensor-like objects."""
TensorType = TypeVar("TensorType", Tensor, np.ndarray, DataFrame, Series)
r"""TypeVar for tensor-like objects."""
CLIPPING_MODE: TypeAlias = Literal["mask", "clip"]
r"""Type Hint for clipping mode."""


def get_broadcast(
    original_shape: tuple[int, ...],
    /,
    *,
    axis: None | int | tuple[int, ...],
    keep_axis: bool = False,
) -> tuple[slice | None, ...]:
    r"""Creates an indexer that broadcasts a tensors contracted via ``axis``.

    achieves: ``data + x[broadcast]`` compatibility.

    Consider a contraction of data along axis, such as `mean` or `sum`.
    This function return an indexer such that the contracted tensor can be
    element-wise operated with the original tensor, by expanding the contracted
    tensor to the shape of.

    >>> arr = np.random.randn(1, 2, 3, 4, 5)
    >>> axis = (1, -1)
    >>> broadcast = get_broadcast(arr.shape, axis=axis)
    >>> m = np.mean(arr, axis)
    >>> m_ref = np.mean(arr, axis=axis, keepdims=True)
    >>> m[broadcast].shape  # == m_ref.shape
    (1, 1, 3, 4, 1)

    Args:
        data: The tensor to be contracted.
        axis: The axes to be contracted.
        keep_axis: select `True` if axis are the axes to be kept instead.

    Example:
        data is of shape  ``(2,3,4,5,6,7)``
        axis is the tuple ``(0,2,-1)``
        broadcast is ``(:, None, :, None, None, :)``
    """
    # selection = [(a % rank) for a in self.axis]
    # caxes = tuple(k for k in range(rank) if k not in selection)
    # broadcast = get_broadcast(data, axis=caxes)
    rank = len(original_shape)

    # determine contraction axes (caxes)
    match axis:
        case None:
            contracted_axes = tuple(range(rank))
        case int():
            contracted_axes = (axis % rank,)
        case _:
            contracted_axes = tuple(a % rank for a in axis)

    if keep_axis:
        return tuple(slice(None) if a in contracted_axes else None for a in range(rank))

    return tuple(None if a in contracted_axes else slice(None) for a in range(rank))


SINGLE_INDEXER: TypeAlias = None | int | list[int] | slice | EllipsisType
r"""Type Hint for single indexer."""
INDEXER: TypeAlias = SINGLE_INDEXER | tuple[SINGLE_INDEXER, ...]
r"""Type Hint for indexer objects."""
AXES: TypeAlias = None | int | tuple[int, ...]
"""Type Hine for axes objects."""


def slice_size(slc: slice) -> int | None:
    if slc.stop is None or slc.start is None:
        return None
    return slc.stop - slc.start


@overload
def get_reduced_axes(item: INDEXER, axes: None) -> None:
    ...


@overload
def get_reduced_axes(item: INDEXER, axes: int | tuple[int, ...]) -> tuple[int, ...]:
    ...


def get_reduced_axes(item: INDEXER, axes: AXES) -> AXES:
    """Determine if a slice would remove some axes."""
    if axes is None:
        return None
    if isinstance(axes, int):
        axes = (axes,)
    if len(axes) == 0:
        return axes

    match item:
        case int():
            return axes[1:]
        case EllipsisType():
            return axes
        case None:
            raise NotImplementedError("Slicing with None not implemented.")
        case list() as lst:
            if len(lst) <= 1:
                return axes[1:]
            return axes
        case slice() as slc:
            if slice_size(slc) in (0, 1):
                return axes[1:]
            return axes
        case tuple() as tup:
            if sum(x is Ellipsis for x in tup) > 1:
                raise ValueError("Only one Ellipsis is allowed.")
            if len(tup) == 0:
                return axes
            if Ellipsis in tup:
                idx = tup.index(Ellipsis)
                return (
                    get_reduced_axes(tup[:idx], axes[:idx])
                    + get_reduced_axes(tup[idx], axes[idx : idx + len(tup) - 1])
                    + get_reduced_axes(tup[idx + 1 :], axes[idx + len(tup) - 1 :])
                )
            return get_reduced_axes(item[0], axes[:1]) + get_reduced_axes(
                item[1:], axes[1:]
            )
        case _:
            raise TypeError(f"Unknown type {type(item)}")


@dataclass
class BoundaryEncoder(BaseEncoder):
    r"""Clip or mask values outside a given range.

    Args:
        lower_bound: the lower boundary. If not provided, it is determined by the mask/data.
        upper_bound: the upper boundary. If not provided, it is determined by the mask/data.
        lower_included: whether the lower boundary is included in the range.
        upper_included: whether the upper boundary is included in the range.
        mode: one of ``'mask'`` or ``'clip'``, or a tuple of two of them for lower and upper.
            - If `mode='mask'`, then values outside the boundary will be replaced by `NA`.
            - If `mode='clip'`, then values outside the boundary will be clipped to it.
        axis: the axis along which to perform the operation.
        requires_fit: whether the lower/upper bounds/values should be determined by the data.
            - if lower/upper not provided, then they are determined by the data.
            - if lower_substitute/upper_substitute not provided, then they are determined by the data.

    Examples:
        - `BoundaryEncoder()` will mask values outside the range `[data_min, data_max]`
        - `BoundaryEncoder(mode='clip')` will clip values to the range `[data_min, data_max]`
        - `BoundaryEncoder(0, 1)` will mask values outside the range `[0,1]`
        - `BoundaryEncoder(0, 1, mode='clip')` will clip values to the range `[0,1]`
        - `BoundaryEncoder(0, 1, mode=('mask', 'clip'))` will mask values below 0 and clip values above 1 to 1.
        - `BoundaryEncoder(0, mode=('mask', 'clip'))` will mask values below 0 and clip values above 1 to `data_max`.
    """

    lower_bound: float | np.ndarray = NotImplemented
    upper_bound: float | np.ndarray = NotImplemented

    _: KW_ONLY

    lower_included: bool = True
    upper_included: bool = True

    mode: CLIPPING_MODE | tuple[CLIPPING_MODE, CLIPPING_MODE] = "mask"

    axis: int | tuple[int, ...] = -1

    requires_fit: bool = field(default=True, init=False, repr=True)
    upper_value: float | np.ndarray = field(
        default=NotImplemented, init=False, repr=False
    )
    lower_value: float | np.ndarray = field(
        default=NotImplemented, init=False, repr=False
    )

    @classmethod
    def from_interval(cls, interval: pd.Interval, **kwargs: Any) -> Self:
        r"""Create a BoundaryEncoder from a pandas Interval."""
        lower_bound = interval.left
        upper_bound = interval.right
        lower_included, upper_included = {
            "left": (True, False),
            "right": (False, True),
            "both": (True, True),
            "neither": (False, False),
        }[interval.closed]
        return cls(
            lower_bound,
            upper_bound,
            lower_included=lower_included,
            upper_included=upper_included,
            **kwargs,
        )

    def __post_init__(self) -> None:
        self.lower_compare = operator.ge if self.lower_included else operator.gt
        self.upper_compare = operator.le if self.upper_included else operator.lt

        match self.mode:
            case "clip":
                mode = ("clip", "clip")
            case "mask":
                mode = ("mask", "mask")
            case "mask" | "clip", "mask" | "clip":
                mode = self.mode
            case _:
                raise ValueError(f"Invalid mode: {self.mode}")

        match mode[0]:
            case "clip":
                if self.lower_bound is not NotImplemented:
                    self.lower_value = self.lower_bound
            case "mask":
                if self.lower_bound is not NotImplemented:
                    raise ValueError("If mode is 'mask', no bound can be provided.")
                self.lower_value = float("nan")
            case _:
                raise ValueError(f"Invalid mode: {mode[0]}")

        match mode[1]:
            case "clip":
                if self.upper_bound is not NotImplemented:
                    self.upper_value = self.upper_bound
            case "mask":
                if self.upper_bound is not NotImplemented:
                    raise ValueError("If mode is 'mask', no bound can be provided.")
                self.upper_value = float("nan")
            case _:
                raise ValueError(f"Invalid mode: {mode[1]}")

        self.requires_fit = (self.lower_bound is NotImplemented) or (
            self.upper_bound is NotImplemented
        )

    def fit(self, data: DataFrame) -> None:
        # TODO: make _nan adapt to real data type!
        if self.requires_fit:
            if self.lower_value is NotImplemented:
                self.lower_value = data.min()
            if self.upper_value is NotImplemented:
                self.upper_value = data.max()

        assert self.lower_bound is not NotImplemented
        assert self.upper_bound is not NotImplemented
        # if isinstance(data, Series | DataFrame) and pd.isna(self.mask_value):
        #     self.mask_value = NA

    def encode(self, data: DataFrame) -> DataFrame:
        # NOTE: frame.where(cond, other) replaces with other if condition is false!
        data = data.where(self.lower_compare(data, self.lower_bound), self.lower_value)
        data = data.where(self.upper_compare(data, self.lower_bound), self.upper_value)
        return data

    def decode(self, data: DataFrame) -> DataFrame:
        return data


@dataclass(init=False)
class LinearScaler(BaseEncoder, Generic[TensorType]):
    r"""Maps the data linearly $x ↦ σ⋅x + μ$.

    Args:
        loc: the offset.
        scale: the scaling factor.
        axis: the axis along which to perform the operation. both μ and σ must have
            shapes that can be broadcasted to the shape of the data along these axis.
    """

    loc: TensorType  # NDArray[np.number] | Tensor
    scale: TensorType  # NDArray[np.number] | Tensor
    r"""The scaling factor."""

    axis: tuple[int, ...]
    r"""Over which axis to perform the scaling."""

    requires_fit: bool = False

    class Parameters(NamedTuple):
        r"""The parameters of the LinearScaler."""

        loc: TensorLike
        scale: TensorLike
        axis: tuple[int, ...]
        requires_fit: bool

        def __repr__(self) -> str:
            r"""Pretty print."""
            return repr_namedtuple(self)

    @property
    def params(self) -> Parameters:
        r"""Parameters of the LinearScaler."""
        return self.Parameters(
            loc=self.loc,
            scale=self.scale,
            axis=self.axis,
            requires_fit=self.requires_fit,
        )

    def __init__(
        self,
        loc: float | TensorType = 0.0,
        scale: float | TensorType = 1.0,
        *,
        axis: None | int | tuple[int, ...] = None,
    ):
        r"""Initialize the MinMaxScaler."""
        super().__init__()
        self.loc = cast(TensorType, loc)
        self.scale = cast(TensorType, scale)
        self.axis = axis  # type: ignore[assignment]

    def __getitem__(self, item: int | slice | tuple[int | slice, ...]) -> Self:
        r"""Return a slice of the LinearScaler.

        Args:
            item: the slice, which is taken directly from the parameters.
                If the parameters are scalars, then we return the same scaler.
                E.g. taking slice encoder[:5] when loc and scale are scalars simply returns the same encoder.
                However, encoder[5] will have to modify the axis, since now this new encoder will only operate
                on the 5th-entry along the first axis.

        Examples:
            - axis is (-2, -1) and data shape is (10, 20, 30, 40). Then
              loc/scale must be broadcastable to (30, 40), i.e. allowed shapes are
              (), (1,), (1,1), (30,), (30,1), (1,40), (30,40).
        """
        # slice the parameters
        loc = self.loc if self.loc.ndim == 0 else self.loc[item]
        scale = self.scale if self.scale.ndim == 0 else self.scale[item]
        axis = get_reduced_axes(item, self.axis)

        # initialize the new encoder
        cls = type(self)
        encoder = cls(loc, scale, axis=axis)
        encoder._is_fitted = self._is_fitted
        return encoder

    def __repr__(self) -> str:
        r"""Pretty print."""
        return repr_dataclass(self)

    def fit(self, data: TensorType, /) -> None:
        if isinstance(data, Tensor):
            self._fit_torch(data)
        else:
            self._fit_numpy(data)

    def _fit_torch(self: LinearScaler[Tensor], data: Tensor) -> None:
        # fix data type
        self.loc = torch.tensor(self.loc, dtype=data.dtype, device=data.device)
        self.scale = torch.tensor(self.scale, dtype=data.dtype, device=data.device)

    def _fit_numpy(self: LinearScaler[np.ndarray], data: NDArray) -> None:
        # fix data type
        self.loc = np.array(self.loc, dtype=data.dtype)
        self.scale = np.array(self.scale, dtype=data.dtype)

    def encode(self, data: TensorType, /) -> TensorType:
        broadcast = get_broadcast(data.shape, axis=self.axis)
        loc = self.loc[broadcast] if self.loc.ndim > 0 else self.loc
        scale = self.scale[broadcast] if self.scale.ndim > 0 else self.scale
        return data * scale + loc

    def decode(self, data: TensorType, /) -> TensorType:
        broadcast = get_broadcast(data.shape, axis=self.axis)
        loc = self.loc[broadcast] if self.loc.ndim > 0 else self.loc
        scale = self.scale[broadcast] if self.scale.ndim > 0 else self.scale
        return (data - loc) / scale


@dataclass(init=False)
class StandardScaler(BaseEncoder, Generic[TensorType]):
    r"""Transforms data linearly x ↦ (x-μ)/σ.

    axis: tuple[int, ...] determines the shape of the mean and stdv.
    """

    mean: TensorType
    r"""The mean value."""
    stdv: TensorType
    r"""The standard-deviation."""
    axis: tuple[int, ...]
    r"""The axis to perform the scaling. If None, automatically select the axis."""
    requires_fit: bool = True

    class Parameters(NamedTuple):
        r"""The parameters of the StandardScalar."""

        mean: TensorLike
        stdv: TensorLike
        axis: None | tuple[int, ...]
        requires_fit: bool

        def __repr__(self) -> str:
            r"""Pretty print."""
            return repr_namedtuple(self)

    @property
    def params(self) -> Parameters:
        r"""Parameters of the Standardizer."""
        return self.Parameters(
            mean=self.mean,
            stdv=self.stdv,
            axis=self.axis,
            requires_fit=self.requires_fit,
        )

    def __init__(
        self,
        /,
        mean: float | TensorType = NotImplemented,
        stdv: float | TensorType = NotImplemented,
        *,
        axis: None | int | tuple[int, ...] = NotImplemented,
    ) -> None:
        super().__init__()
        self.mean = cast(TensorType, mean)
        self.stdv = cast(TensorType, stdv)
        self.axis = axis  # type: ignore[assignment]
        self.requires_fit = (mean is NotImplemented) or (stdv is NotImplemented)

    def __repr__(self) -> str:
        r"""Pretty print."""
        return repr_dataclass(self)

    def __getitem__(self, item: int | slice | list[int]) -> Self:
        r"""Return a slice of the Standardizer."""
        # slice the parameters
        mean = self.mean[item] if self.mean.ndim > 0 else self.mean
        stdv = self.stdv[item] if self.stdv.ndim > 0 else self.stdv
        axis = get_reduced_axes(item, self.axis)

        # initialize the new encoder
        cls = type(self)
        encoder = cls(mean=mean, stdv=stdv, axis=axis)
        encoder._is_fitted = self._is_fitted
        return encoder

    def fit(self, data: TensorType, /) -> None:
        if not self.requires_fit:
            return

        rank = len(data.shape)

        # set the axis
        if self.axis is None:
            self.axis = ()
        elif isinstance(self.axis, int):
            self.axis = (self.axis,)
        elif self.axis is NotImplemented:
            self.axis = (-1,) if len(data.shape) > 1 else ()
        else:
            self.axis = tuple(self.axis)

        # determine the axes to perform contraction over (caxes).
        selection = {(a % rank) for a in self.axis}
        caxes = tuple(k for k in range(rank) if k not in selection)
        broadcast = get_broadcast(data.shape, axis=caxes)

        # determine the backend
        backend: ModuleType
        as_tensor: Callable[[Any], TensorType]

        if isinstance(data, Tensor):
            backend = torch
            as_tensor = torch.tensor
        else:
            backend = np
            as_tensor = np.array

        # compute the mean
        if self.mean is NotImplemented:
            self.mean = backend.nanmean(data, axis=caxes)
        self.mean = as_tensor(self.mean)

        # compute the standard deviation
        if self.stdv is NotImplemented:
            mean = self.mean[broadcast]
            # print(f"{data.ndim=}  {caxes=}, {broadcast=}  {mean=}  {mean.ndim=}")
            self.stdv = backend.sqrt(backend.nanmean((data - mean) ** 2, axis=caxes))
        self.stdv = as_tensor(self.stdv)

    def encode(self, data: TensorType, /) -> TensorType:
        broadcast = get_broadcast(data.shape, axis=self.axis, keep_axis=True)
        return (data - self.mean[broadcast]) / self.stdv[broadcast]

    def decode(self, data: TensorType, /) -> TensorType:
        broadcast = get_broadcast(data.shape, axis=self.axis, keep_axis=True)
        return data * self.stdv[broadcast] + self.mean[broadcast]


@dataclass(init=False)
class MinMaxScaler(BaseEncoder, Generic[TensorType]):
    r"""Linearly transforms [x_min, x_max] to [y_min, y_max] (default: [0, 1])."""

    ymin: TensorType  # NDArray[np.number] | Tensor
    ymax: TensorType  # NDArray[np.number] | Tensor
    xmin: TensorType  # NDArray[np.number] | Tensor
    xmax: TensorType  # NDArray[np.number] | Tensor
    scale: TensorType  # NDArray[np.number] | Tensor
    r"""The scaling factor."""
    axis: tuple[int, ...]
    r"""Over which axis to perform the scaling."""

    requires_fit: bool = True

    class Parameters(NamedTuple):
        r"""The parameters of the MinMaxScaler."""

        xmin: TensorLike
        xmax: TensorLike
        ymin: TensorLike
        ymax: TensorLike
        scale: TensorLike
        axis: tuple[int, ...]
        requires_fit: bool

        def __repr__(self) -> str:
            r"""Pretty print."""
            return repr_namedtuple(self)

    @property
    def params(self) -> Parameters:
        r"""Parameters of the MinMaxScaler."""
        return self.Parameters(
            xmin=self.xmin,
            xmax=self.xmax,
            ymin=self.ymin,
            ymax=self.ymax,
            scale=self.scale,
            axis=self.axis,
            requires_fit=self.requires_fit,
        )

    def __init__(
        self,
        ymin: float | TensorType = 0.0,
        ymax: float | TensorType = 1.0,
        *,
        xmin: float | TensorType = 0.0,
        xmax: float | TensorType = 1.0,
        axis: None | int | tuple[int, ...] = NotImplemented,
        requires_fit: bool = True,
    ):
        r"""Initialize the MinMaxScaler."""
        super().__init__()
        self.xmin = cast(TensorType, np.array(xmin))
        self.xmax = cast(TensorType, np.array(xmax))
        self.ymin = cast(TensorType, np.array(ymin))
        self.ymax = cast(TensorType, np.array(ymax))
        self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)
        self.axis = axis  # type: ignore[assignment]
        self.requires_fit = requires_fit

    def __getitem__(self, item: int | slice | list[int]) -> Self:
        r"""Return a slice of the MinMaxScaler."""
        # slice the parameters
        xmin = self.xmin[item] if self.xmin.ndim > 0 else self.xmin
        xmax = self.xmax[item] if self.xmax.ndim > 0 else self.xmax
        ymin = self.ymin[item] if self.ymin.ndim > 0 else self.ymin
        ymax = self.ymax[item] if self.ymax.ndim > 0 else self.ymax
        axis = get_reduced_axes(item, self.axis)

        # initialize the new encoder
        cls = type(self)
        encoder = cls(ymin, ymax, xmin=xmin, xmax=xmax, axis=axis)
        encoder._is_fitted = self._is_fitted
        return encoder

    def __repr__(self) -> str:
        r"""Pretty print."""
        return repr_dataclass(self)

    def fit(self, data: TensorType, /) -> None:
        # TODO: Why does singledispatch not work here? (wrap_func in BaseEncoder)
        # print(type(data), isinstance(data, np.ndarray), isinstance(data, type))
        if not self.requires_fit:
            return

        # set the axis
        if self.axis is None:
            self.axis = ()
        elif isinstance(self.axis, int):
            self.axis = (self.axis,)
        elif self.axis is NotImplemented:
            self.axis = (-1,) if len(data.shape) > 1 else ()
        else:
            self.axis = tuple(self.axis)

        if isinstance(data, Tensor):
            self._fit_torch(data)
        else:
            self._fit_numpy(np.asarray(data))

    def _fit_torch(self: MinMaxScaler[Tensor], data: Tensor, /) -> None:
        r"""Compute the min and max."""
        rank = len(data.shape)
        selection = {(a % rank) for a in self.axis}
        axes = tuple(k for k in range(rank) if k not in selection)

        mask = torch.isnan(data)
        neginf = torch.tensor(float("-inf"), device=data.device, dtype=data.dtype)
        posinf = torch.tensor(float("+inf"), device=data.device, dtype=data.dtype)

        self.xmin = torch.amin(torch.where(mask, posinf, data), dim=axes)
        self.xmax = torch.amax(torch.where(mask, neginf, data), dim=axes)

        self.ymin = torch.tensor(self.ymin, device=data.device, dtype=data.dtype)
        self.ymax = torch.tensor(self.ymax, device=data.device, dtype=data.dtype)
        # broadcast y to the same shape as x
        self.ymin = self.ymin + torch.zeros_like(self.xmin)
        self.ymax = self.ymax + torch.zeros_like(self.xmax)

        self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)

    def _fit_numpy(self: MinMaxScaler[np.ndarray], data: NDArray, /) -> None:
        r"""Compute the min and max."""
        rank = len(data.shape)
        selection = {(a % rank) for a in self.axis}
        axes = tuple(k for k in range(rank) if k not in selection)

        self.xmin = np.nanmin(data, axis=axes)
        self.xmax = np.nanmax(data, axis=axes)

        self.ymin = np.array(self.ymin, dtype=data.dtype)
        self.ymax = np.array(self.ymax, dtype=data.dtype)

        # broadcast y to the same shape as x
        self.ymin = self.ymin + np.zeros_like(self.xmin)
        self.ymax = self.ymax + np.zeros_like(self.xmax)

        self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)

    def encode(self, data: TensorType, /) -> TensorType:
        broadcast = get_broadcast(data.shape, axis=self.axis, keep_axis=True)
        xmin = self.xmin[broadcast]  # if self.xmin.ndim > 1 else self.xmin
        scale = self.scale[broadcast]  # if self.scale.ndim > 1 else self.scale
        ymin = self.ymin[broadcast]  # if self.ymin.ndim > 1 else self.ymin
        return (data - xmin) * scale + ymin

    def decode(self, data: TensorType, /) -> TensorType:
        broadcast = get_broadcast(data.shape, axis=self.axis, keep_axis=True)
        xmin = self.xmin[broadcast]  # if self.xmin.ndim > 1 else self.xmin
        scale = self.scale[broadcast]  # if self.scale.ndim > 1 else self.scale
        ymin = self.ymin[broadcast]  # if self.ymin.ndim > 1 else self.ymin
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

    def decode(self, data: PandasObject, /) -> PandasObject:
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
