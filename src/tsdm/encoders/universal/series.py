r"""Linear data encoders."""

__all__ = [
    # Classes
    "LinearScaler",
    "MinMaxScaler",
    "StandardScaler",
]

from dataclasses import KW_ONLY, dataclass, field
from typing import Any, cast

from tsdm.backend import Backend, get_backend
from tsdm.constants import UNDEFINED
from tsdm.encoders.base import FittableEncoder, StaticEncoder
from tsdm.pprint import pprint_repr

type FloatArray = Any


@pprint_repr
@dataclass(frozen=True, slots=True)
class LinearScaler[T: FloatArray](StaticEncoder[T, T]):
    r"""Maps the data linearly $x ↦ σ⋅x + μ$.

    Args:
        loc: the offset.
        scale: the scaling factor.
    """

    loc: float = 0.0
    scale: float = 1.0

    def encode[S: FloatArray](self, data: S, /) -> S:
        return cast("Any", data) * self.scale + self.loc

    def decode[S: FloatArray](self, data: S, /) -> S:
        return (cast("Any", data) - self.loc) / self.scale


@pprint_repr
@dataclass(slots=True)
class StandardScaler[T: FloatArray](FittableEncoder[T, T]):
    r"""Transforms data linearly $x ↦ (x-μ)/σ$."""

    mean: float = UNDEFINED
    r"""The mean value."""
    stdv: float = UNDEFINED
    r"""The standard-deviation."""

    def fit(self, data: Any, /) -> None:
        # switch the backend
        backend: Backend = get_backend(data)

        self.mean = float(backend.nanmean(data))
        self.stdv = float(backend.nanstd(data))

    def encode[S: FloatArray](self, data: S, /) -> S:
        return (cast("Any", data) - self.mean) / self.stdv

    def decode[S: FloatArray](self, data: S, /) -> S:
        return cast("Any", data) * self.stdv + self.mean


@pprint_repr
@dataclass(slots=True)
class MinMaxScaler[T: FloatArray](FittableEncoder[T, T]):
    r"""Linearly transforms [x_min, x_max] to [y_min, y_max] (default: [0, 1]).

    If x_min and/or x_max are provided at initialization, they are marked as
    "fitted" and will not be re-computed during the fit method.

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

        In particular, when $xₘᵢₙ = xₘₐₓ$, we have $x̄ = xₘᵢₙ = xₘₐₓ$, and formally set
        the scale to 1, making the transform $x ↦ x + (ȳ - x̄)$
    """

    ymin: float = 0.0
    ymax: float = 1.0

    _: KW_ONLY

    xmin: float = UNDEFINED  # or ScalarType.
    xmax: float = UNDEFINED  # or ScalarType.

    safe_computation: bool = True
    r"""Whether to ensure that the bounds are not violated due to roundoff."""

    xmin_learnable: bool = field(init=False)
    xmax_learnable: bool = field(init=False)

    xbar: float = field(init=False, default=UNDEFINED)
    ybar: float = field(init=False, default=UNDEFINED)
    scale: float = field(init=False, default=UNDEFINED)

    def __post_init__(self) -> None:
        assert self.ymin is not UNDEFINED
        assert self.ymax is not UNDEFINED
        self.ybar = (self.ymax + self.ymin) / 2
        self.xmin_learnable = self.xmin is UNDEFINED
        self.xmax_learnable = self.xmax is UNDEFINED

    def fit(self, data: Any, /) -> None:
        backend = get_backend(data)

        if self.xmin_learnable:
            self.xmin = float(backend.nanmin(data))
        if self.xmax_learnable:
            self.xmax = float(backend.nanmax(data))

        self.xbar = (self.xmax + self.xmin) / 2
        self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)

    def encode[S: FloatArray](self, x: S, /) -> S:
        r"""Maps [xₘᵢₙ, xₘₐₓ] to [yₘᵢₙ, yₘₐₓ]."""
        x_any = cast("Any", x)
        y = (x_any - self.xbar) * self.scale + self.ybar
        if self.safe_computation:
            # x < x_min, set y to y_min, x > x_max, set y to y_max
            backend = get_backend(x_any)
            y = backend.where(x_any < self.xmin, self.ymin, y)
            y = backend.where(x_any > self.xmax, self.ymax, y)
            y = backend.clip(y, self.ymin, self.ymax)
        return cast("S", y)

    def decode[S: FloatArray](self, y: S, /) -> S:
        r"""Maps [yₘᵢₙ, yₘₐₓ] to [xₘᵢₙ, xₘₐₓ]."""
        y_any = cast("Any", y)
        x = (y_any - self.ybar) / self.scale + self.xbar
        if self.safe_computation:
            # y < y_min, set x to x_min, y > y_max, set x to x_max
            backend = get_backend(y_any)
            x = backend.where(y_any < self.ymin, self.xmin, x)
            x = backend.where(y_any > self.ymax, self.xmax, x)
            x = backend.clip(x, self.xmin, self.xmax)
        return cast("S", x)
