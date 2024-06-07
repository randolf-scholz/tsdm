r"""Numerical Transformations, Standardization, Log-Transforms, etc.

Numerical Encoders should be able to be applied with different backends such as

- numpy arrays
- pandas dataframes
- torch tensors
- pyarrow tables
- etc.

To ensure performance during encoding/decoding, the backend should be fixed.

Note:
    Golden Rule for implementation: init/fit can be slow, but transform should be fast.
    Use specialization/dispatch to ensure fast transforms.

Goals
=====
- numerical encoders should allow for different backends: numpy, pandas, torch, etc.
- numerical encoders should be vectorized and fast
- we should be able to "slice" vectorized encoders just like we slice numpy arrays
- calling fit twice on the same data should not change the encoder (idempotent)
- one should be able to (partially or fully) fix the encoder parameters
- should have an axis attribute that allows for broadcasting.
- switching between backends should be easy and fast
    - switching between backends probably not considered a "fit" operation
    - fitting changes the encoder parameter values, switching backends changes their types.
"""

__all__ = [
    # Protocols & ABCs
    "NumericalEncoder",
    # Functions
    "get_broadcast",
    "get_reduced_axes",
    "invert_axis_selection",
    # Classes
    "BoundaryEncoder",
    "LinearScaler",
    "LogEncoder",
    "LogitEncoder",
    "MinMaxScaler",
    "StandardScaler",
    "TensorConcatenator",
    "TensorSplitter",
    # Functions
    "slice_size",
]

from collections.abc import Iterable
from dataclasses import KW_ONLY, asdict, dataclass, field
from types import EllipsisType

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pandas import DataFrame, Series
from torch import Tensor
from typing_extensions import (
    Any,
    Literal,
    Optional,
    Self,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

from tsdm.backend import Backend, get_backend
from tsdm.encoders.base import BaseEncoder
from tsdm.types.aliases import Axis, Nested, Size
from tsdm.types.protocols import NTuple, NumericalArray
from tsdm.utils.decorators import pprint_repr

Arr = TypeVar("Arr", bound=NumericalArray)
r"""TypeVar for tensor-like objects."""
Arr2 = TypeVar("Arr2", bound=NumericalArray)
r"""TypeVar for tensor-like objects."""
Index: TypeAlias = None | int | list[int] | slice | EllipsisType
r"""Type Hint for single indexer."""
Scalar: TypeAlias = None | bool | int | float | complex | str
r"""Type Hint for scalar objects."""
ClippingMode: TypeAlias = Literal["mask", "clip"]
r"""Type Hint for clipping mode."""
TensorType = TypeVar("TensorType", Tensor, NDArray)
r"""TypeVar for tensor-like objects."""

PARAMETERS: TypeAlias = tuple[
    Scalar
    | Arr
    | list[Scalar]
    | list[Arr]
    | list["PARAMETERS"]
    | tuple[Scalar | Arr | "PARAMETERS", ...]
    | dict[str, Scalar]
    | dict[str, Arr]
    | dict[str, "PARAMETERS"],
    ...,
]
r"""Type Hint for parameters tuple."""


def invert_axis_selection(axis: Axis, /, *, ndim: int) -> tuple[int, ...]:
    r"""Invert axes-selection for a rank `ndim` tensor.

    Example:
        +------+------------+-----------+
        | ndim | axis       | inverted  |
        +======+============+===========+
        | 4    | None       | ()        |
        +------+------------+-----------+
        | 4    | (-3,-2,-1) | (0,)      |
        +------+------------+-----------+
        | 4    | (-2,-1)    | (0,1)     |
        +------+------------+-----------+
        | 4    | (-1)       | (0,1,2)   |
        +------+------------+-----------+
        | 4    | ()         | (0,1,2,3) |
        +------+------------+-----------+
    """
    match axis:
        case None:
            return ()
        case int():
            return tuple(set(range(ndim)) - {axis % ndim})
        case Iterable():
            return tuple(set(range(ndim)) - {a % ndim for a in axis})
        case _:
            raise TypeError(f"axis must be None, int, or Iterable, not {type(axis)}")


def get_broadcast(
    original_shape: tuple[int, ...],
    /,
    *,
    axis: Axis,
    keep_axis: bool = False,
) -> tuple[slice | None, ...]:
    r"""Creates an indexer that broadcasts a tensors contracted via `axis`.

    Essentially works like a-posteriori adding the `keepdims=True` option to a contraction.
    If `x = contraction(data, axis)` (e.g. sum, mean, max), then ``x[broadcast]``
    for ``broadcast=get_braodcast(data.shape, axis) is roughly equivalent to
    ``contraction(data, axis, keepdims=True)``.

    This achieves element-wise compatibility: ``data + x[broadcast]``.

    >>> arr = np.random.randn(1, 2, 3, 4, 5)
    >>> axis = (1, -1)
    >>> broadcast = get_broadcast(arr.shape, axis=axis)
    >>> m = np.mean(arr, axis)
    >>> m_ref = np.mean(arr, axis=axis, keepdims=True)
    >>> m[broadcast].shape == m_ref.shape
    True

    If `keep_axis` is True, then the broadcast is the complement of the contraction,
    i.e. ``x[broadcast]`` is roughly equivalent to ``contraction(data, kept_axis, keepdims=True)``,
    where ``kept_axis = set(range(data.ndim)) - set(ax%data.ndim for ax in axis)``.

    Args:
        original_shape: The tensor to be contracted.
        axis: The axes to be contracted.
        keep_axis: select `True` if the axes are to be kept instead.

    Example:
        data is of shape  ``(2,3,4,5,6,7)``
        axis is the tuple ``(0,2,-1)``
        broadcast is ``(:, None, :, None, None, :)``
    """
    rank = len(original_shape)

    if keep_axis:
        match axis:
            case None:  # all axes are contracted
                kept_axis = set()
            case int():
                kept_axis = {axis % rank}
            case _:
                kept_axis = {a % rank for a in axis}
        return tuple(slice(None) if a in kept_axis else None for a in range(rank))

    match axis:
        case None:  # all axes are contracted
            contracted_axes = set(range(rank))
        case int():
            contracted_axes = {axis % rank}
        case _:
            contracted_axes = {a % rank for a in axis}

    return tuple(None if a in contracted_axes else slice(None) for a in range(rank))


def slice_size(slc: slice, /) -> Optional[int]:
    r"""Get the size of a slice."""
    if slc.stop is None or slc.start is None:
        return None
    return slc.stop - slc.start


@overload
def get_reduced_axes(item: Index | tuple[Index, ...], axis: None) -> None: ...
@overload
def get_reduced_axes(
    item: Index | tuple[Index, ...], axis: Size
) -> tuple[int, ...]: ...
def get_reduced_axes(item, axis):
    r"""Determine if a slice would remove some axes."""
    match axis:
        case None:
            return None
        case []:
            return ()  # type: ignore[unreachable]
        case int(a):
            axis = (a,)
        case _:
            axis = tuple(axis)

    match item:
        case int():
            return axis[1:]
        case EllipsisType():
            return axis
        case None:
            raise NotImplementedError("Slicing with None not implemented.")
        case list() as lst:
            if len(lst) <= 1:
                return axis[1:]
            return axis
        case slice() as slc:
            if slice_size(slc) in {0, 1}:
                return axis[1:]
            return axis
        case tuple() as tup:
            if sum(x is Ellipsis for x in tup) > 1:
                raise ValueError("Only one Ellipsis is allowed.")
            if len(tup) == 0:
                return axis
            if Ellipsis in tup:
                idx = tup.index(Ellipsis)
                return (
                    get_reduced_axes(tup[:idx], axis[:idx])
                    + get_reduced_axes(tup[idx], axis[idx : idx + len(tup) - 1])
                    + get_reduced_axes(tup[idx + 1 :], axis[idx + len(tup) - 1 :])
                )
            return get_reduced_axes(item[0], axis[:1]) + get_reduced_axes(
                item[1:], axis[1:]
            )
        case _:
            raise TypeError(f"Unknown type {type(item)}")


class NumericalEncoder(BaseEncoder[Arr, Arr]):
    r"""Represents a numerical encoder."""

    backend: Backend[Arr]
    r"""The backend of the encoder."""

    def switch_backend(self, backend: str) -> None:
        r"""Switch the backend of the encoder."""
        self.backend: Backend[Arr] = Backend(backend)

        # recast the parameters
        self.recast_parameters()

    def recast_parameters(self) -> None:
        r"""Recast the parameters to the current backend."""
        # switch the backend of the parameters
        self.cast_params(self.params)
        raise NotImplementedError

    def cast_params(self, params: Nested) -> Nested[Arr]:
        r"""Cast the parameters to the current backend."""
        match params:
            case NTuple() as ntup:
                cls = type(ntup)
                return cls(*(self.cast_params(p) for p in ntup))
            case tuple() as tup:
                return tuple(self.cast_params(p) for p in tup)
            case list() as lst:
                return [self.cast_params(p) for p in lst]
            case dict() as dct:
                return {k: self.cast_params(v) for k, v in dct.items()}
            case Tensor() | np.ndarray() | Series() | DataFrame() as t:
                return self.backend.to_tensor(t)
            case _ as p:
                return p


@pprint_repr
@dataclass
class BoundaryEncoder(BaseEncoder[Arr, Arr]):
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


    Note:
        requires_fit: whether the data should determine the lower/upper bounds/values.
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

    lower_bound: None | float | Arr = NotImplemented
    upper_bound: None | float | Arr = NotImplemented

    _: KW_ONLY

    axis: Axis = None
    lower_included: bool = True
    upper_included: bool = True
    mode: ClippingMode | tuple[ClippingMode, ClippingMode] = "mask"

    # derived attributes
    backend: Backend[Arr] = field(init=False)
    lower_value: float | Arr = field(init=False, default=NotImplemented)
    upper_value: float | Arr = field(init=False, default=NotImplemented)

    @property
    def params(self) -> dict[str, Any]:
        return asdict(self)

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

    @property
    def lower_mode(self) -> ClippingMode:
        r"""The mode for the lower boundary."""
        return self.mode[0] if isinstance(self.mode, tuple) else self.mode

    @property
    def upper_mode(self) -> ClippingMode:
        r"""The mode for the upper boundary."""
        return self.mode[1] if isinstance(self.mode, tuple) else self.mode

    def __post_init__(self) -> None:
        if pd.isna(self.lower_bound):
            self.lower_bound = None

        if pd.isna(self.upper_bound):
            self.upper_bound = None

        if (
            self.upper_bound is not None
            and self.upper_bound is not NotImplemented
            and self.lower_bound is not None
            and self.lower_bound is not NotImplemented
            and self.upper_bound <= self.lower_bound
        ):
            raise ValueError("lower_bound must be smaller than upper_bound.")

    def lower_satisfied(self, x: Arr) -> Arr:
        r"""Return a boolean mask for the lower boundary (true: value ok)."""
        if self.lower_bound is None:
            return self.backend.true_like(x)
        if self.lower_included or self.lower_included is None:
            return (x >= self.lower_bound) | self.backend.isnan(x)
        return (x > self.lower_bound) | self.backend.isnan(x)

    def upper_satisfied(self, x: Arr) -> Arr:
        r"""Return a boolean mask for the upper boundary (true: value ok)."""
        if self.upper_bound is None:
            return self.backend.true_like(x)
        if self.upper_included or self.upper_included is None:
            return (x <= self.upper_bound) | self.backend.isnan(x)
        return (x < self.upper_bound) | self.backend.isnan(x)

    def encode(self, data: Arr, /) -> Arr:
        # NOTE: frame.where(cond, other) replaces with other if condition is false!
        data = self.backend.where(self.lower_satisfied(data), data, self.lower_value)
        data = self.backend.where(self.upper_satisfied(data), data, self.upper_value)
        return data

    def decode(self, data: Arr, /) -> Arr:
        return data

    def fit(self, data: Arr) -> None:
        # select the backend
        selected_backend = get_backend(data)
        self.backend: Backend[Arr] = Backend(selected_backend)

        # fit the parameters
        if self.lower_bound is NotImplemented:
            self.lower_bound = self.backend.nanmin(data)
        if self.upper_bound is NotImplemented:
            self.upper_bound = self.backend.nanmax(data)

        # set lower_value
        match self.lower_bound, self.lower_mode:
            case None, _:
                self.lower_value = self.backend.to_tensor(float("-inf"))
            case _, "mask":
                self.lower_value = self.backend.to_tensor(float("nan"))
            case _, "clip":
                self.lower_value = self.lower_bound  # type: ignore[assignment]
            case _:
                raise NotImplementedError

        # set upper_value
        match self.upper_bound, self.upper_mode:
            case None, _:
                self.upper_value = self.backend.to_tensor(float("+inf"))
            case _, "mask":
                self.upper_value = self.backend.to_tensor(float("nan"))
            case _, "clip":
                self.upper_value = self.upper_bound  # type: ignore[assignment]
            case _:
                raise NotImplementedError


@pprint_repr
@dataclass(init=False)
class LinearScaler(BaseEncoder[Arr, Arr]):
    r"""Maps the data linearly $x ↦ σ⋅x + μ$.

    Args:
        loc: the offset.
        scale: the scaling factor.
        axis: the axis along which to perform the operation. both μ and σ must have
            shapes that can be broadcasted to the shape of the data along these axes.
    """

    loc: Arr  # NDArray[np.number] | Tensor
    scale: Arr  # NDArray[np.number] | Tensor
    r"""The scaling factor."""

    axis: Axis
    r"""Over which axis to perform the scaling."""
    backend: Backend[Arr]
    r"""The backend of the encoder."""

    @property
    def params(self) -> dict[str, Any]:
        return asdict(self)

    def __init__(
        self,
        loc: float | Arr = 0.0,
        scale: float | Arr = 1.0,
        *,
        axis: Axis = None,
    ) -> None:
        r"""Initialize the MinMaxScaler."""
        self.loc = cast(Arr, loc)
        self.scale = cast(Arr, scale)
        self.axis = axis

        if axis is not None:
            raise NotImplementedError("Axis not implemented yet.")

    def __getitem__(self, item: int | slice | tuple[int | slice, ...], /) -> Self:
        r"""Return a slice of the LinearScaler.

        Args:
            item: the slice, which is taken directly from the parameters.
                If the parameters are scalars, then we return the same scaler.
                E.g. taking slice encoder[:5] when loc and scale are scalars simply returns the same encoder.
                However, encoder[5] will have to modify the axis, since the new encoder will only operate
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

    def fit(self, data: Arr, /) -> None:
        selected_backend = get_backend(data)
        self.backend: Backend[Arr] = Backend(selected_backend)

    def encode(self, data: Arr, /) -> Arr:
        return data * self.scale + self.loc

    def decode(self, data: Arr, /) -> Arr:
        return (data - self.loc) / self.scale


@dataclass(init=False)
class StandardScaler(BaseEncoder[Arr, Arr]):
    r"""Transforms data linearly x ↦ (x-μ)/σ.

    axis: tuple[int, ...] determines the shape of the mean and stdv.
    """

    mean: Arr
    r"""The mean value."""
    stdv: Arr
    r"""The standard-deviation."""

    _: KW_ONLY

    axis: Axis
    r"""The axis to perform the scaling. If None, automatically select the axis."""
    backend: Backend[Arr] = NotImplemented
    r"""The backend of the encoder."""

    @property
    def params(self) -> dict[str, Any]:
        return asdict(self)

    def __init__(
        self,
        mean: float | Arr = NotImplemented,
        stdv: float | Arr = NotImplemented,
        *,
        axis: Axis = (),
    ) -> None:
        self.mean = cast(Arr, mean)
        self.stdv = cast(Arr, stdv)
        self.axis = axis
        self.mean_learnable = mean is NotImplemented
        self.stdv_learnable = stdv is NotImplemented

    def __getitem__(self, item: int | slice | list[int], /) -> Self:
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

    def fit(self, data: Arr, /) -> None:
        # switch the backend
        selected_backend = get_backend(data)
        self.backend: Backend[Arr] = Backend(selected_backend)

        # universal fitting procedure
        axes = invert_axis_selection(self.axis, ndim=len(data.shape))

        if self.mean_learnable:
            self.mean = self.backend.nanmean(data, axis=axes)

        if self.stdv_learnable:
            self.stdv = self.backend.nanstd(data, axis=axes)

    def encode(self, data: Arr, /) -> Arr:
        # TODO: consider adding broadcasting
        #   1. broadcast = get_broadcast(data.shape, axis=self.axis, keep_axis=True)
        #   2. return (data - self.mean[broadcast]) / self.stdv[broadcast]
        return (data - self.mean) / self.stdv

    def decode(self, data: Arr, /) -> Arr:
        # TODO: consider adding broadcasting
        #   1. broadcast = get_broadcast(data.shape, axis=self.axis, keep_axis=True)
        #   2. return data * self.stdv[broadcast] + self.mean[broadcast]
        return data * self.stdv + self.mean


@dataclass(init=False)
class MinMaxScaler(BaseEncoder[Arr, Arr]):
    r"""Linearly transforms [x_min, x_max] to [y_min, y_max] (default: [0, 1]).

    If x_min and/or x_max are provided at initialization, they are marked as
    "fitted" and will not be re-computed during the fit method.

    Examples:
        - axis=() (default): fit one scaler for all elements.
        - axis=-1: fit one scaler per channel.
        - axis=(-2, -1): fit one scaler per image.
        - axis=None: fit one scaler per element.

    Note:
        In the edge case when fitting to a single value, we set the scale such that
        the encoded value is ½(y_min + y_max).

    Note:
        Generally, the transformation is given by the formula:

        .. math:: x ↦ \frac{x - xₘᵢₙ}{xₘₐₓ - xₘᵢₙ}(yₘₐₓ - yₘᵢₙ) + yₘᵢₙ

        We transform the formula, to cater to the edge case, by extending with the
        average of the min and max x-values:

        .. math::
            \frac{yₘₐₓ - yₘᵢₙ}{xₘₐₓ - xₘᵢₙ}(x - xₘᵢₙ) + yₘᵢₙ \\
            = \frac{x +½(xₘₐₓ - xₘᵢₙ) -½(xₘₐₓ - xₘᵢₙ) - xₘᵢₙ}{xₘₐₓ - xₘᵢₙ}(yₘₐₓ - yₘᵢₙ) + yₘᵢₙ \\
            = ½(yₘₐₓ - yₘᵢₙ) + \frac{x - ½(xₘₐₓ + xₘᵢₙ)}{xₘₐₓ - xₘᵢₙ}(yₘₐₓ - yₘᵢₙ) + yₘᵢₙ \\
            = \frac{yₘₐₓ - yₘᵢₙ}{xₘₐₓ - xₘᵢₙ}(x - x̄) + ȳ \\
            = γ⋅(x - x̄) + ȳ

        In particular, when xₘᵢₙ = xₘₐₓ, we have x̄ = xₘᵢₙ = xₘₐₓ, and formally set
        the scale to 1, making the transform x ↦ x + (ȳ - x̄)

    Note:
        Whatever formula we use, it should be ensured that when x∈[x_min, x_max],
        then y∈[y_min, y_max], i.e. the result should be within bounds.
        This might be violated due to numerical roundoff, so we need to be careful.
    """

    ymin: Arr  # or: ScalarType.
    ymax: Arr  # or: ScalarType.
    xmin: Arr  # or: ScalarType.
    xmax: Arr  # or: ScalarType.
    scale: Arr  # or: ScalarType.
    r"""The scaling factor."""

    _: KW_ONLY

    axis: Axis
    r"""Over which axis to perform the scaling."""
    safe_computation: bool
    r"""Whether to ensure that the bounds are not violated due to roundoff."""
    backend: Backend[Arr] = NotImplemented
    r"""The backend of the encoder."""

    @property
    def params(self) -> dict[str, Any]:
        return asdict(self)

    def __init__(
        self,
        ymin: float | Arr = 0.0,
        ymax: float | Arr = 1.0,
        *,
        xmin: None | float | Arr = None,
        xmax: None | float | Arr = None,
        axis: Axis = (),
    ) -> None:
        self.safe_computation = True
        self.ymin = cast(Arr, ymin)
        self.ymax = cast(Arr, ymax)
        self.axis = axis

        self.xmin_learnable = xmin is None
        self.xmax_learnable = xmax is None
        self.xmin = cast(Arr, NotImplemented if xmin is None else xmin)
        self.xmax = cast(Arr, NotImplemented if xmax is None else xmax)

        # set derived parameters
        if not (self.xmin_learnable or self.xmax_learnable):
            self.xbar: Arr = (self.xmax + self.xmin) / 2
            self.ybar: Arr = (self.ymax + self.ymin) / 2
            self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)
        else:
            self.xbar = NotImplemented
            self.ybar = NotImplemented
            self.scale = NotImplemented

        # set initial backend
        self.switch_backend(get_backend(self.params))

    def __getitem__(self, item: int | slice | list[int], /) -> Self:
        r"""Return a slice of the MinMaxScaler."""
        # slice the parameters
        xmin = self.xmin[item] if self.xmin.ndim > 0 else self.xmin
        xmax = self.xmax[item] if self.xmax.ndim > 0 else self.xmax
        ymin = self.ymin[item] if self.ymin.ndim > 0 else self.ymin
        ymax = self.ymax[item] if self.ymax.ndim > 0 else self.ymax
        axis = get_reduced_axes(item, self.axis)

        # initialize the new encoder
        cls: type[Self] = type(self)
        encoder = cls(ymin, ymax, xmin=xmin, xmax=xmax, axis=axis)
        encoder.switch_backend(self.backend)
        encoder._is_fitted = self._is_fitted
        return encoder

    def fit(self, data: Arr, /) -> None:
        # switch the backend
        self.switch_backend(get_backend(data))

        # skip if the parameters are not learnable
        if not (self.xmin_learnable or self.xmax_learnable):
            return

        # universal fitting procedure
        axes = invert_axis_selection(self.axis, ndim=len(data.shape))

        if self.xmin_learnable:
            self.xmin = self.backend.nanmin(data, axis=axes)
        if self.xmax_learnable:
            self.xmax = self.backend.nanmax(data, axis=axes)

        # broadcast y to the same shape as x
        self.ymin = self.ymin + 0.0 * self.xmin
        self.ymax = self.ymax + 0.0 * self.xmax

        # update the average variables
        self.xbar = (self.xmax + self.xmin) / 2
        self.ybar = (self.ymax + self.ymin) / 2

        # compute the scale
        dx = self.xmax - self.xmin
        dy = self.ymax - self.ymin
        scale = dy / dx
        self.scale = self.backend.where(dx != 0, scale, scale**0)

    def encode(self, x: Arr, /) -> Arr:
        r"""Maps [xₘᵢₙ, xₘₐₓ] to [yₘᵢₙ, yₘₐₓ]."""
        # TODO: consider adding broadcasting

        # unpacking for easier readability
        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax
        xbar = self.xbar
        ybar = self.ybar
        scale = self.scale

        y = (x - xbar) * scale + ybar

        if self.safe_computation:
            # NOTE: ensures the conditions
            #   x < xₘᵢₙ ⟹ y < yₘᵢₙ  ∧  x > xₘₐₓ ⟹ y > yₘₐₓ  ∧  x∈[xₘᵢₙ, xₘₐₓ] ⟹ y∈[yₘᵢₙ, yₘₐₓ]
            y = self.backend.where(x < xmin, self.backend.clip(y, None, ymin), y)
            y = self.backend.where(x > xmax, self.backend.clip(y, ymax, None), y)
            y = self.backend.where(
                (x >= xmin) & (x <= xmax), self.backend.clip(y, ymin, ymax), y
            )

        return y

    def decode(self, y: Arr, /) -> Arr:
        r"""Maps [yₘᵢₙ, yₘₐₓ] to [xₘᵢₙ, xₘₐₓ]."""
        # TODO: consider adding broadcasting

        # unpacking for easier readability
        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax
        xbar = self.xbar
        ybar = self.ybar
        scale = self.scale

        x = (y - ybar) / scale + xbar

        if self.safe_computation:
            # NOTE: ensures the conditions
            #   y < yₘᵢₙ ⟹ x < xₘᵢₙ  ∧  y > yₘₐₓ ⟹ x > xₘₐₓ  ∧  y∈[yₘᵢₙ, yₘₐₓ] ⟹ x∈[xₘᵢₙ, xₘₐₓ]
            x = self.backend.where(y < ymin, self.backend.clip(x, None, xmin), x)
            x = self.backend.where(y > ymax, self.backend.clip(x, xmax, None), x)
            x = self.backend.where(
                (y >= ymin) & (y <= ymax), self.backend.clip(x, xmin, xmax), x
            )

        return x

    # region parameters ----------------------------------------------------------------

    def recompute_params(self):
        r"""Computes derived parameters from the base parameters."""
        self.xbar = (self.xmax + self.xmin) / 2
        self.ybar = (self.ymax + self.ymin) / 2
        self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)

    def switch_backend(self, backend: str | Backend) -> None:
        r"""Switch the backend of the scaler."""
        self.backend: Backend[Arr] = Backend(backend)
        self.recast_parameters()

    def recast_parameters(self) -> None:
        r"""Recast the parameters to the current backend."""
        # switch the backend of the parameters
        self.xmin = self.backend.to_tensor(self.xmin)
        self.xmax = self.backend.to_tensor(self.xmax)
        self.ymin = self.backend.to_tensor(self.ymin)
        self.ymax = self.backend.to_tensor(self.ymax)
        self.xbar = self.backend.to_tensor(self.xbar)
        self.ybar = self.backend.to_tensor(self.ybar)
        self.scale = self.backend.to_tensor(self.scale)

    # endregion parameters -------------------------------------------------------------


@dataclass
class LogEncoder(BaseEncoder[NDArray, NDArray]):
    r"""Encode data on a logarithmic scale.

    Uses base 2 by default for lower numerical error and fast computation.
    """

    threshold: NDArray
    replacement: NDArray

    @property
    def params(self) -> dict[str, Any]:
        return asdict(self)

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


class LogitEncoder(BaseEncoder[NDArray, NDArray]):
    r"""Logit encoder."""

    @property
    def params(self) -> dict[str, Any]:
        return {}

    def encode(self, data: DataFrame, /) -> DataFrame:
        assert all((data > 0) & (data < 1))
        return np.log(data / (1 - data))

    def decode(self, data: DataFrame, /) -> DataFrame:
        return np.clip(1 / (1 + np.exp(-data)), 0, 1)


class TensorSplitter(BaseEncoder[TensorType, list[TensorType]]):
    r"""Split tensor along specified axis."""

    lengths: list[int]
    numdims: list[int]
    axis: int
    maxdim: int
    indices_or_sections: int | list[int]

    @property
    def params(self) -> dict[str, Any]:
        return {"indices_or_sections": self.indices_or_sections, "axis": self.axis}

    def __init__(
        self, *, indices_or_sections: int | list[int] = 1, axis: int = 0
    ) -> None:
        r"""Concatenate tensors along the specified axis."""
        self.axis = axis
        self.indices_or_sections = indices_or_sections

    def encode(self, data: TensorType, /) -> list[TensorType]:
        if isinstance(data, Tensor):
            return list(
                torch.tensor_split(data, self.indices_or_sections, dim=self.axis)
            )
        return np.array_split(data, self.indices_or_sections, dim=self.axis)  # type: ignore[call-overload]

    def decode(self, data: list[TensorType], /) -> TensorType:
        if isinstance(data[0], Tensor):
            return torch.cat(data, dim=self.axis)
        return np.concatenate(data, axis=self.axis)


class TensorConcatenator(BaseEncoder[list[Tensor], Tensor]):
    r"""Concatenate multiple tensors.

    Useful for concatenating encoders for multiple inputs.
    """

    lengths: list[int]
    numdims: list[int]
    axis: int
    maxdim: int

    @property
    def params(self) -> dict[str, Any]:
        return {"axis": self.axis}

    def __init__(self, axis: int = 0) -> None:
        r"""Concatenate tensors along the specified axis."""
        self.axis = axis

    def fit(self, data: list[Tensor], /) -> None:
        self.numdims = [d.ndim for d in data]
        self.maxdim = max(self.numdims)
        # pad dimensions if necessary
        arrays = [d[(...,) + (None,) * (self.maxdim - d.ndim)] for d in data]
        # store the lengths of the slices
        self.lengths = [x.shape[self.axis] for x in arrays]

    def encode(self, data: list[Tensor], /) -> Tensor:
        return torch.cat(
            [d[(...,) + (None,) * (self.maxdim - d.ndim)] for d in data], dim=self.axis
        )

    def decode(self, data: Tensor, /) -> list[Tensor]:
        result = torch.split(data, self.lengths, dim=self.axis)
        return [x.squeeze() for x in result]
