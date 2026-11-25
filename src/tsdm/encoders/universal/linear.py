r"""Linear data encoders."""

__all__ = [
    # Classes
    "LinearScaler",
    "MinMaxScaler",
    "StandardScaler",
    # functions
    "invert_axis_selection",
    "reduce_axes",
]

from dataclasses import KW_ONLY, dataclass, field
from typing import Any, cast, overload

from tsdm.backend import get_backend
from tsdm.constants import UNDEFINED
from tsdm.encoders.base import FittableEncoder, StaticEncoder
from tsdm.types import Axis, IndexArgND
from tsdm.utils import pretty_repr

type FloatArray = Any


def invert_axis_selection(axis: Axis, /, *, ndim: int) -> tuple[int, ...]:
    r"""Return the complement of an axis selection for a tensor of rank `ndim`.

    Negative axes are normalized modulo `ndim`. An `axis` of `None` selects
    every axis, so its complement is empty. The output order is unspecified.

    The following examples show the complementary axes in sorted order:

    | `ndim` | `axis`         | Complement     |
    |--------|----------------|----------------|
    | 4      | `None`         | `()`           |
    | 4      | `(-3, -2, -1)` | `(0,)`         |
    | 4      | `(-2, -1)`     | `(0, 1)`       |
    | 4      | `(-1,)`        | `(0, 1, 2)`    |
    | 4      | `()`           | `(0, 1, 2, 3)` |
    """
    match axis:
        case None:
            return ()
        case int():
            return tuple(set(range(ndim)) - {axis % ndim})
        case tuple():
            return tuple(set(range(ndim)) - {a % ndim for a in axis})
        case _:
            raise TypeError(f"axis must be None, int, or Iterable, not {type(axis)}")


@overload
def reduce_axes(axis: None, selection: IndexArgND) -> None: ...
@overload
def reduce_axes(
    axis: int | tuple[int, ...], selection: str | list[str] | IndexArgND
) -> tuple[int, ...]: ...
def reduce_axes(
    axis: Axis, selection: str | list[str] | IndexArgND
) -> tuple[int, ...] | None:
    r"""Reduce an axis selection to account for tensor indexing.

    Given an operation `op(T, axis=axis)`, use
    `reduced_axis = reduce_axes(axis, selection)` to obtain the axis selection
    for `op(T[selection], axis=reduced_axis)`.

    Slice indices preserve axes, including when they select zero or one entry.
    An `axis` of `None` remains `None`. Inserting dimensions with `None` is
    unsupported for explicit axis selections; at most one ellipsis is allowed.
    """
    # convert to tuple
    match axis:
        case None:
            return None
        case []:
            return ()
        case int(a):
            axis = (a,)
        case _:
            axis = tuple(axis)

    match selection:
        case None:
            raise NotImplementedError("Slicing with None not implemented.")
        case int() | str():
            return axis[1:]
        case slice():
            return axis
        case ellipsis if ellipsis is ...:
            return axis
        case list(seq):
            drop = len(seq) <= 1
            return axis[drop:]
        case range(start=start, stop=stop):
            drop = (stop - start) in {0, 1}
            return axis[drop:]
        case tuple(tup):
            if sum(x is Ellipsis for x in tup) > 1:
                raise ValueError("Only one Ellipsis is allowed.")
            if len(tup) == 0:
                return axis
            if Ellipsis in tup:
                idx = tup.index(Ellipsis)
                return (
                    reduce_axes(axis[:idx], tup[:idx])
                    + reduce_axes(axis[idx : idx + len(tup) - 1], tup[idx])
                    + reduce_axes(axis[idx + len(tup) - 1 :], tup[idx + 1 :])
                )
            # recurse on the first element
            return reduce_axes(axis[:1], tup[0]) + reduce_axes(axis[1:], tup[1:])
        case _:
            raise TypeError(f"Unknown type {type(selection)}")


def _reduce_param(param: Any, selection: Any) -> Any:
    r"""Select parameter entries while leaving scalar parameters unchanged.

    Array parameters are indexed with `selection`. Python scalars and
    zero-dimensional arrays are returned unchanged.
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


@pretty_repr
@dataclass(slots=True, frozen=True)
class LinearScaler[Arr: FloatArray](StaticEncoder[Arr, Arr]):
    r"""Map data linearly as $x ↦ σ⋅x + μ$.

    Args:
        loc: The offset $μ$, broadcastable to the input shape.
        scale: The factor $σ$, broadcastable to the input shape.
        axis: Must be `None`; explicit axis selection is not implemented.
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
        r"""Return a scaler with sliced parameters.

        Args:
            item: Selection applied to array-valued `loc` and `scale` parameters.
                Scalar parameters remain unchanged. The returned scaler must still
                satisfy the constructor's axis restrictions.
        """
        axis = reduce_axes(self.axis, item)
        loc = _reduce_param(self.loc, item)
        scale = _reduce_param(self.scale, item)
        return LinearScaler(loc=loc, scale=scale, axis=axis)

    def encode(self, data: Arr, /) -> Arr:
        return cast("Arr", cast("Any", data) * self.scale + self.loc)

    def decode(self, data: Arr, /) -> Arr:
        return cast("Arr", (cast("Any", data) - self.loc) / self.scale)


@pretty_repr
@dataclass(slots=True)
class StandardScaler[Arr: FloatArray](FittableEncoder[Arr, Arr]):
    r"""Transform data linearly as $x ↦ (x-μ)/σ$.

    `axis` specifies the axes retained in the fitted mean and standard deviation.
    The remaining axes are reduced. `axis=()` fits global statistics; `axis=None`
    retains every axis. The fitted parameters must broadcast to the input shape.
    """

    mean: Any = UNDEFINED
    r"""The mean value."""
    stdv: Any = UNDEFINED
    r"""The standard-deviation."""

    _: KW_ONLY

    axis: Axis = ()
    r"""Axes retained in the fitted statistics. `None` retains all axes."""

    mean_learnable: bool = field(init=False)
    stdv_learnable: bool = field(init=False)

    def __post_init__(self) -> None:
        self.mean_learnable = self.mean is UNDEFINED
        self.stdv_learnable = self.stdv is UNDEFINED

    def __getitem__(self, item: Any, /) -> StandardScaler[Arr]:
        r"""Return a slice of the StandardScaler."""
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


@pretty_repr
@dataclass(slots=True)
class MinMaxScaler[Arr: FloatArray](FittableEncoder[Arr, Arr]):
    r"""Map $[xₘᵢₙ, xₘₐₓ]$ linearly to $[yₘᵢₙ, yₘₐₓ]$ (default: $[0, 1]$).

    If `xmin` and/or `xmax` are provided at initialization, they are marked as
    "fitted" and will not be re-computed during the fit method.

    Examples:
        - axis=() (default): fit one scaler for all elements.
        - axis=-1: fit one scaler per channel.
        - axis=(-2, -1): fit one scaler per image.
        - axis=None: fit one scaler per element.

    Note:
        In the edge case when fitting to a single value, we set the scale such that
        the encoded value is $½(yₘᵢₙ + yₘₐₓ)$.

    Note:
        Generally, the transformation is given by the formula:

        $$ x ↦ \frac{x - xₘᵢₙ}{xₘₐₓ - xₘᵢₙ}(yₘₐₓ - yₘᵢₙ) + yₘᵢₙ $$

        Centering at the input and output midpoints gives an equivalent formula:

        $$\begin{aligned}
            y &= \frac{yₘₐₓ - yₘᵢₙ}{xₘₐₓ - xₘᵢₙ}(x - x̄) + ȳ \\
            x̄ &= ½(xₘₐₓ + xₘᵢₙ) \\
            ȳ &= ½(yₘₐₓ + yₘᵢₙ)
        \end{aligned}$$

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
