r"""Linear data encoders."""

__all__ = [
    # Classes
    "LinearScaler",
    "MinMaxScaler",
    "StandardScaler",
]

from dataclasses import KW_ONLY, dataclass, field
from typing import Any, cast

from tsdm.backend import get_backend
from tsdm.constants import UNDEFINED
from tsdm.encoders.base import FittableEncoder, StaticEncoder
from tsdm.linalg.utils import invert_axis_selection, reduce_axes
from tsdm.pprint import pprint_repr
from tsdm.types.aliases import Axis

type FloatArray = Any


def _reduce_param(param: Any, selection: Any) -> Any:
    r"""Perform a reduction on a parameter.

    For example, given tensor T, axis and selection, then this returns the slice of the tensor
    that satisfies the selection.
    """
    match param:
        case (int() | float()) as scalar:
            # NOTE: need to test int | float because of typing issues.
            # FIXME: https://github.com/python/typing/issues/1746
            return scalar
        case scalar if len(scalar.shape) == 0:
            return scalar
        case tensor:
            return tensor[selection]


@pprint_repr
@dataclass(frozen=True, slots=True)
class LinearScaler[Arr: FloatArray](StaticEncoder[Arr, Arr]):
    r"""Maps the data linearly $x ↦ σ⋅x + μ$.

    Args:
        loc: the offset.
        scale: the scaling factor.
        axis: the axis along which to perform the operation. both μ and σ must have
            shapes that can be broadcasted to the shape of the data along these axes.
    """

    loc: Any = 0.0
    scale: Any = 1.0
    r"""The scaling factor."""

    _: KW_ONLY

    axis: Axis = None
    r"""Over which axis to perform the scaling."""

    def __post_init__(self) -> None:
        if self.axis is not None:
            raise NotImplementedError("Axis not implemented yet.")

    def __getitem__(self, item: Any, /) -> LinearScaler[Arr]:
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
        axis = reduce_axes(self.axis, item)
        loc = _reduce_param(self.loc, item)
        scale = _reduce_param(self.scale, item)
        return LinearScaler(loc=loc, scale=scale, axis=axis)

    def encode(self, data: Arr, /) -> Arr:
        return cast("Arr", cast("Any", data) * self.scale + self.loc)

    def decode(self, data: Arr, /) -> Arr:
        return cast("Arr", (cast("Any", data) - self.loc) / self.scale)


@pprint_repr
@dataclass(slots=True)
class StandardScaler[Arr: FloatArray](FittableEncoder[Arr, Arr]):
    r"""Transforms data linearly x ↦ (x-μ)/σ.

    axis: tuple[int, ...] determines the shape of the mean and stdv.
    """

    mean: Any = UNDEFINED
    r"""The mean value."""
    stdv: Any = UNDEFINED
    r"""The standard-deviation."""

    _: KW_ONLY

    axis: Axis = ()
    r"""The axis to perform the scaling. If None, automatically select the axis."""

    mean_learnable: bool = field(init=False)
    stdv_learnable: bool = field(init=False)

    def __post_init__(self) -> None:
        self.mean_learnable = self.mean is UNDEFINED
        self.stdv_learnable = self.stdv is UNDEFINED

    def __getitem__(self, item: Any, /) -> StandardScaler[Arr]:
        r"""Return a slice of the Standardizer."""
        mean = _reduce_param(self.mean, item)
        stdv = _reduce_param(self.stdv, item)
        axis = reduce_axes(self.axis, item)
        return StandardScaler(mean=mean, stdv=stdv, axis=axis)

    def fit(self, data: Arr, /) -> None:
        backend = get_backend(data)

        axes = invert_axis_selection(self.axis, ndim=len(cast("Any", data).shape))

        if self.mean_learnable:
            self.mean = backend.nanmean(data, axis=axes)

        if self.stdv_learnable:
            self.stdv = backend.nanstd(data, axis=axes)

    def encode(self, data: Arr, /) -> Arr:
        # TODO: consider adding broadcasting
        #   1. broadcast = get_broadcast(data.shape, axis=self.axis, keep_axis=True)
        #   2. return (data - self.mean[broadcast]) / self.stdv[broadcast]
        return cast("Arr", (cast("Any", data) - self.mean) / self.stdv)

    def decode(self, data: Arr, /) -> Arr:
        # TODO: consider adding broadcasting
        #   1. broadcast = get_broadcast(data.shape, axis=self.axis, keep_axis=True)
        #   2. return data * self.stdv[broadcast] + self.mean[broadcast]
        return cast("Arr", cast("Any", data) * self.stdv + self.mean)


@pprint_repr
@dataclass(slots=True)
class MinMaxScaler[Arr: FloatArray](FittableEncoder[Arr, Arr]):
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

    ymin: Any = 0.0
    ymax: Any = 1.0

    _: KW_ONLY

    xmin: Any = UNDEFINED
    xmax: Any = UNDEFINED

    axis: Axis = ()
    r"""Over which axis to perform the scaling."""
    safe_computation: bool = True
    r"""Whether to ensure that the bounds are not violated due to roundoff."""

    xbar: Any = field(init=False, default=UNDEFINED)
    ybar: Any = field(init=False, default=UNDEFINED)
    scale: Any = field(init=False, default=UNDEFINED)
    xmin_learnable: bool = field(init=False)
    xmax_learnable: bool = field(init=False)

    def __post_init__(self) -> None:
        assert self.ymin is not UNDEFINED
        assert self.ymax is not UNDEFINED
        self.xmin_learnable = self.xmin is UNDEFINED
        self.xmax_learnable = self.xmax is UNDEFINED

        if not self.xmin_learnable and not self.xmax_learnable:
            self.ymin = self.ymin + 0.0 * self.xmin
            self.ymax = self.ymax + 0.0 * self.xmax
            self.ybar = (self.ymax + self.ymin) / 2
            self.xbar = (self.xmax + self.xmin) / 2
            self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)

    def __getitem__(self, item: Any, /) -> MinMaxScaler[Arr]:
        r"""Return a slice of the MinMaxScaler."""
        xmin = _reduce_param(self.xmin, item)
        xmax = _reduce_param(self.xmax, item)
        ymin = _reduce_param(self.ymin, item)
        ymax = _reduce_param(self.ymax, item)
        axis = reduce_axes(self.axis, item)
        return MinMaxScaler(ymin, ymax, xmin=xmin, xmax=xmax, axis=axis)

    def fit(self, data: Arr, /) -> None:
        backend = get_backend(data)

        # invert axes selection
        axes = invert_axis_selection(self.axis, ndim=len(cast("Any", data).shape))
        if self.xmin_learnable:
            self.xmin = backend.nanmin(data, axis=axes)
        if self.xmax_learnable:
            self.xmax = backend.nanmax(data, axis=axes)

        # broadcast y to the same shape as x
        self.ymin = self.ymin + 0.0 * self.xmin
        self.ymax = self.ymax + 0.0 * self.xmax
        self.ybar = (self.ymax + self.ymin) / 2
        self.xbar = (self.xmax + self.xmin) / 2
        dx = self.xmax - self.xmin
        dy = self.ymax - self.ymin
        scale = dy / dx
        self.scale = backend.where(dx != 0, scale, scale**0)

    def encode(self, x: Arr, /) -> Arr:
        r"""Maps [xₘᵢₙ, xₘₐₓ] to [yₘᵢₙ, yₘₐₓ]."""
        x_any = cast("Any", x)
        y = (x_any - self.xbar) * self.scale + self.ybar
        if self.safe_computation:
            # x < x_min, set y to y_min, x > x_max, set y to y_max
            backend = get_backend(x)
            y = backend.where(x_any < self.xmin, self.ymin, y)
            y = backend.where(x_any > self.xmax, self.ymax, y)
            y = backend.clip(y, self.ymin, self.ymax)
        return cast("Arr", y)

    def decode(self, y: Arr, /) -> Arr:
        r"""Maps [yₘᵢₙ, yₘₐₓ] to [xₘᵢₙ, xₘₐₓ]."""
        y_any = cast("Any", y)
        x = (y_any - self.ybar) / self.scale + self.xbar
        if self.safe_computation:
            # y < y_min, set x to x_min, y > y_max, set x to x_max
            backend = get_backend(y)
            x = backend.where(y_any < self.ymin, self.xmin, x)
            x = backend.where(y_any > self.ymax, self.xmax, x)
            x = backend.clip(x, self.xmin, self.xmax)
        return cast("Arr", x)
