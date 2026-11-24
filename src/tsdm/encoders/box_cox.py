r"""Box-Cox encoder."""

__all__ = [
    # Classes
    "BoxCoxEncoder",
    "LogitBoxCoxEncoder",
    "LogEncoder",
    "LogitEncoder",
    # Functions
    "construct_wasserstein_loss_boxcox_uniform",
    "construct_wasserstein_loss_boxcox_normal",
    "construct_wasserstein_loss_logit_uniform",
    "construct_wasserstein_loss_logit_normal",
]

import warnings
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from enum import StrEnum
from typing import Final, assert_never

import numpy as np
from numpy import pi as PI
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import erfinv

from tsdm.constants import UNDEFINED
from tsdm.types import SupportsArrayUfunc
from tsdm.utils import pretty_repr

from .base import FittableEncoder, StaticEncoder

_ROOT_3: Final[float] = 1.7320508075688772  # sqrt(3)


def construct_wasserstein_loss_boxcox_uniform(
    x: NDArray,
    /,
    *,
    lower: float = -_ROOT_3,
    upper: float = +_ROOT_3,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    r"""Construct a squared Wasserstein-2 loss against $p = \op{Uniform}([a, b])$.

    $$ ℒ(c) = W₂²(p, q̃_c^\emp) = ∫₀¹ \abs{F⁻¹(t) - G_c⁻¹(t)}² \dd{t} $$

    After excluding NaNs, let $x₁ < ⋯ < x_K$ be the distinct observations with
    relative frequencies $αₖ = nₖ/N$ and cumulative masses $tₖ = ∑ⱼ₌₁ᵏ αⱼ$.
    The empirical measure is $q^\emp = ∑ₖ αₖ δ_{xₖ}$, where $δ_v$ is a unit point
    mass at $v$. The implementation transforms it to $q̃_c^\emp = ∑ₖ αₖ δ_{zₖ(c)}$:

    $$\begin{aligned}
        uₖ(c) &= \log(xₖ+c) \\
        zₖ(c) &= \frac{σ}{s(c)}(uₖ(c)-m(c)+μ) \\
        m(c) &= K⁻¹∑ₖ uₖ(c)        &  s(c)² &= K⁻¹∑ₖ \abs{uₖ(c)-m(c)}²
    \end{aligned}$$

    The moments weight unique values equally, although the loss retains their
    frequencies. Also, adding $μ$ before scaling gives mean $μσ/s(c)$ over unique
    values, rather than $μ$; repeated observations can further change the moments.

    Here, $F⁻¹$ and $G_c⁻¹$ are the target and transformed empirical quantiles.
    Since $G_c⁻¹(t) = zₖ(c)$ on $(tₖ₋₁, tₖ]$, splitting the integral gives

    $$\begin{aligned}
        βₖ &= ∫_{tₖ₋₁}^{tₖ} F⁻¹(t) \dd{t}, & Bₖ &= -2βₖ/αₖ \\
        C &= ∫₀¹ F⁻¹(t)² \dd{t}, & ℒ(c) &= ∑ₖ αₖ[zₖ(c)²+Bₖzₖ(c)+C]
    \end{aligned}$$

    For the uniform target, $H′ = F⁻¹$ and the coefficients are

    $$\begin{aligned}
        F⁻¹(t) &= a+(b-a)t, & H(t) &= at+½(b-a)t² \\
        βₖ &= H(tₖ)-H(tₖ₋₁), & C &= ⅓(a²+ab+b²) = μ²+σ² \\
        μ &= ½(a+b), & σ &= (b-a)/\sqrt{12} \\
        a &= μ-\sqrt{3}σ, & b &= μ+\sqrt{3}σ
    \end{aligned}$$

    Args:
        x: Observations of any shape; NaNs are excluded. At least two distinct,
            finite values are required. Offsets must give finite $\log(xₖ+c)$ and
            $s(c) > 0$, with every $xₖ+c > 0$.
        lower: Target lower endpoint $a$; must be below `upper`.
        upper: Target upper endpoint $b$; defaults give $μ = 0$, $σ = 1$.

    Returns:
        A callable evaluating $ℒ(c)$ elementwise, with the same shape as `c`.
    """

    def integrate_quantile(q: NDArray[np.float64]) -> NDArray[np.float64]:
        if (lower, upper) == (-_ROOT_3, +_ROOT_3):
            return _ROOT_3 * q * (q - 1)
        return lower * q + (upper - lower) * q**2 / 2

    a = lower
    b = upper
    mask = np.isnan(x)
    unique, counts = np.unique(x[~mask], return_counts=True)
    α = counts / np.sum(counts)
    p = np.insert(np.cumsum(α), 0, 0).clip(0, 1)
    β = integrate_quantile(p[1:]) - integrate_quantile(p[:-1])
    μ = (b + a) / 2
    σ = abs(b - a) / np.sqrt(12)
    B = -2 * (β / α)
    C = 1.0 if (a, b) == (-_ROOT_3, +_ROOT_3) else (a**2 + a * b + b**2) / 3

    def fun(c: NDArray[np.float64]) -> NDArray[np.float64]:
        u = np.log(np.add.outer(c, unique))
        # transform to target loc-scale
        mean = np.mean(u, axis=-1, keepdims=True)
        stdv = np.std(u, axis=-1, keepdims=True)
        y = (u - mean + μ) * (σ / stdv)
        return np.einsum("...i, i -> ...", y**2 + B * y + C, α)

    return fun


def construct_wasserstein_loss_boxcox_normal(
    x: NDArray, /, *, loc: float = 0.0, scale: float = 1.0
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    r"""Construct a squared Wasserstein-2 loss against $p = \op{Normal}(μ, σ²)$.

    $$ ℒ(c) = W₂²(p, q̃_c^\emp) = ∫₀¹ [F⁻¹(t)-G_c⁻¹(t)]² \dd{t} $$

    Use the empirical masses, cumulative masses, and normalization from
    [the uniform loss][tsdm.encoders.box_cox.construct_wasserstein_loss_boxcox_uniform]:
    $uₖ(c) = \log(xₖ+c)$ and $zₖ(c) = [uₖ(c)-m(c)+μ]σ/s(c)$.
    The moments $m(c), s(c)$ weight unique values equally; the loss uses observation
    frequencies $αₖ$. The mean over unique values is $μσ/s(c)$, rather than $μ$.

    With $H′ = F⁻¹$, the normal target gives

    $$\begin{aligned}
        F⁻¹(t) &= μ+σ\sqrt{2}\erf⁻¹(2t-1) \\
        h(t) &= \exp\left(-[\erf⁻¹(2t-1)]²\right) \\
        H(t) &= μt-\frac{σ}{\sqrt{2π}}h(t) \\
        βₖ &= H(tₖ)-H(tₖ₋₁), & Bₖ &= -2βₖ/αₖ \\
        C &= μ²+σ², & ℒ(c) &= ∑ₖ αₖ[zₖ(c)²+Bₖzₖ(c)+C]
    \end{aligned}$$

    Here $\erf⁻¹$ is the inverse error function. By continuity, $h(0) = h(1) = 0$,
    so $βₖ$ stays finite despite the divergent endpoint quantiles.

    Args:
        x: Observations of any shape; NaNs are excluded. At least two distinct,
            finite values are required. Offsets must give finite $\log(xₖ+c)$ and
            $s(c) > 0$, with every $xₖ+c > 0$.
        loc: Target mean $μ$; defaults to zero.
        scale: Target standard deviation $σ > 0$; defaults to one.

    Returns:
        A callable evaluating $ℒ(c)$ elementwise, with the same shape as `c`.
    """

    def integrate_quantile(q: NDArray[np.float64]) -> NDArray[np.float64]:
        if (loc, scale) == (0, 1):
            return -np.exp(-(erfinv(2 * q - 1) ** 2)) / np.sqrt(2 * PI)
        return loc * q - scale * np.exp(-(erfinv(2 * q - 1) ** 2)) / np.sqrt(2 * PI)

    μ = loc
    σ = scale
    mask = np.isnan(x)
    unique, counts = np.unique(x[~mask], return_counts=True)
    α = counts / np.sum(counts)
    p = np.insert(np.cumsum(α), 0, 0).clip(0, 1)
    β = integrate_quantile(p[1:]) - integrate_quantile(p[:-1])
    C = μ**2 + σ**2
    B = -2 * (β / α)

    def fun(c: NDArray[np.float64], /) -> NDArray[np.float64]:
        # apply the box-cox transform
        u = np.log(np.add.outer(c, unique))
        # normalize to mean=0, std=1
        mean = np.mean(u, axis=-1, keepdims=True)
        stdv = np.std(u, axis=-1, keepdims=True)
        y = (u - mean + μ) * (σ / stdv)
        # compute the loss
        return np.einsum("...i, i -> ...", y**2 + B * y + C, α)

    return fun


def construct_wasserstein_loss_logit_uniform(
    x: NDArray,
    /,
    *,
    lower: float = -_ROOT_3,
    upper: float = +_ROOT_3,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    r"""Construct a squared Wasserstein-2 loss against $p = \op{Uniform}([a, b])$.

    $$ ℒ(c) = W₂²(p, q̃_c^\emp) = ∫₀¹ [F⁻¹(t)-G_c⁻¹(t)]² \dd{t} $$

    Use the empirical masses, normalization, and uniform coefficients from
    [the Box-Cox loss][tsdm.encoders.box_cox.construct_wasserstein_loss_boxcox_uniform],
    with the offset logit replacing the logarithm:

    $$ uₖ(c) = \log\left(\frac{xₖ+c}{1-xₖ+c}\right) $$

    The implementation uses $zₖ(c) = [uₖ(c)-m(c)+μ]σ/s(c)$, with moments over unique
    values and observation frequencies in the loss. Its mean over unique values
    is $μσ/s(c)$, rather than $μ$; repeated observations can further change moments.

    Args:
        x: Observations of any shape; NaNs are excluded. At least two distinct,
            finite values are required. Offsets must give finite logarithms and
            $s(c) > 0$, with $xₖ+c > 0$ and $1-xₖ+c > 0$. For data in $[0, 1]$,
            any $c > 0$ satisfies the positivity conditions.
        lower: Target lower endpoint $a$; must be below `upper`.
        upper: Target upper endpoint $b$; defaults give $μ = 0$, $σ = 1$.

    Returns:
        A callable evaluating $ℒ(c)$ elementwise, with the same shape as `c`.
    """

    def integrate_quantile(q: NDArray[np.float64]) -> NDArray[np.float64]:
        if (lower, upper) == (-_ROOT_3, +_ROOT_3):
            return _ROOT_3 * q * (q - 1)
        return lower * q + (upper - lower) * q**2 / 2

    a = lower
    b = upper
    mask = np.isnan(x)
    unique, counts = np.unique(x[~mask], return_counts=True)
    α = counts / np.sum(counts)
    p = np.insert(np.cumsum(α), 0, 0).clip(0, 1)
    β = integrate_quantile(p[1:]) - integrate_quantile(p[:-1])
    μ = (b + a) / 2
    σ = abs(b - a) / np.sqrt(12)
    B = -2 * (β / α)
    C = 1.0 if (a, b) == (-_ROOT_3, +_ROOT_3) else (a**2 + a * b + b**2) / 3

    def fun(c: NDArray[np.float64]) -> NDArray[np.float64]:
        u = np.log(np.add.outer(c, unique) / (1 + np.add.outer(c, -unique)))
        # transform to target loc-scale
        mean = np.mean(u, axis=-1, keepdims=True)
        stdv = np.std(u, axis=-1, keepdims=True)
        y = (u - mean + μ) * (σ / stdv)
        return np.einsum("...i, i -> ...", y**2 + B * y + C, α)

    return fun


def construct_wasserstein_loss_logit_normal(
    x: NDArray, /, *, loc: float = 0.0, scale: float = 1.0
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    r"""Construct a squared Wasserstein-2 loss against $p = \op{Normal}(μ, σ²)$.

    $$ ℒ(c) = W₂²(p, q̃_c^\emp) = ∫₀¹ [F⁻¹(t)-G_c⁻¹(t)]² \dd{t} $$

    Use the empirical masses, normalization, and normal coefficients from
    [the Box-Cox loss][tsdm.encoders.box_cox.construct_wasserstein_loss_boxcox_normal],
    with the offset logit replacing the logarithm:

    $$ uₖ(c) = \log\left(\frac{xₖ+c}{1-xₖ+c}\right) $$

    The implementation uses $zₖ(c) = [uₖ(c)-m(c)+μ]σ/s(c)$, with moments over unique
    values and observation frequencies in the loss. Its mean over unique values
    is $μσ/s(c)$, rather than $μ$; repeated observations can further change moments.

    Args:
        x: Observations of any shape; NaNs are excluded. At least two distinct,
            finite values are required. Offsets must give finite logarithms and
            $s(c) > 0$, with $xₖ+c > 0$ and $1-xₖ+c > 0$. For data in $[0, 1]$,
            any $c > 0$ satisfies the positivity conditions.
        loc: Target mean $μ$; defaults to zero.
        scale: Target standard deviation $σ > 0$; defaults to one.

    Returns:
        A callable evaluating $ℒ(c)$ elementwise, with the same shape as `c`.
    """

    def integrate_quantile(q: NDArray[np.float64]) -> NDArray[np.float64]:
        if (loc, scale) == (0, 1):
            return -np.exp(-(erfinv(2 * q - 1) ** 2)) / np.sqrt(2 * PI)
        return loc * q - scale * np.exp(-(erfinv(2 * q - 1) ** 2)) / np.sqrt(2 * PI)

    μ = loc
    σ = scale
    mask = np.isnan(x)
    unique, counts = np.unique(x[~mask], return_counts=True)
    α = counts / np.sum(counts)
    p = np.insert(np.cumsum(α), 0, 0).clip(0, 1)
    β = integrate_quantile(p[1:]) - integrate_quantile(p[:-1])
    B = -2 * (β / α)
    C = 1.0 if (μ, σ) == (0, 1) else μ**2 + σ**2

    def fun(c: NDArray[np.float64]) -> NDArray[np.float64]:
        u = np.log(np.add.outer(c, unique) / (1 + np.add.outer(c, -unique)))
        # transform to target loc-scale
        mean = np.mean(u, axis=-1, keepdims=True)
        stdv = np.std(u, axis=-1, keepdims=True)
        y = (u - mean + μ) * (σ / stdv)
        return np.einsum("...i, i -> ...", y**2 + B * y + C, α)

    return fun


@pretty_repr
@dataclass(slots=True, init=False)
class BoxCoxEncoder[T: SupportsArrayUfunc](FittableEncoder[T, T]):
    r"""Encode nonnegative data with a fitted offset $c$ that makes $x+c > 0$.

    $$ \op{encode}_c(x) = \log(x+c), \qquad x ∈ [0, ∞) $$

    Decoding clips negative reconstructions to zero:

    $$ \op{decode}_c(y) = \max(eʸ-c, 0) $$

    NaNs are excluded during fitting and propagate through encoding and decoding.
    With $Q₁, Q₃$ denoting the non-NaN data's first and third quartiles:

    1. `fixed`: use `offset_guess` as $c$.
    2. `minimum` (default): use $c = ½\min\set{xᵢ : xᵢ > 0}$.
       At least one positive observation is required.
    3. `quartile`: use $c = (Q₁/Q₃)²$. The denominator must be positive.
    4. `match-uniform` or `match-normal`: minimize the squared Wasserstein-2 loss
       of the centered, rescaled data against a uniform or normal target with
       mean zero and variance one.
       At least two distinct, finite observations are required.

    Matching centers and rescales only during fitting; encoding returns $\log(x+c)$.
    """

    class METHOD(StrEnum):
        r"""Methods to fit the Box-Cox parameter."""

        fixed = "fixed"
        minimum = "minimum"
        quartile = "quartile"
        match_normal = "match-normal"
        match_uniform = "match-uniform"

    _: KW_ONLY

    bounds: tuple[float, float] = (0.0, 1.0)
    method: METHOD = METHOD.match_uniform
    offset: float = UNDEFINED
    offset_guess: float = 1.0
    verbose: bool = False

    # FIXME: simplify if a converter option is added to the dataclass.field
    def __init__(
        self,
        *,
        bounds: tuple[float, float] = (0.0, 1.0),
        offset_guess: float = 1.0,
        method: METHOD | str = "minimum",
        offset: float = UNDEFINED,
        verbose: bool = False,
    ) -> None:
        if method not in self.METHOD:
            raise ValueError(f"{method=} unknown. Available: {', '.join(self.METHOD)}")

        self.bounds = bounds
        self.offset_guess = offset_guess
        self.method = self.METHOD(method)
        self.offset = offset
        self.verbose = verbose

        if self.method == self.METHOD.fixed:
            self.offset = self.offset_guess
            self.validate_params()

    def validate_params(self) -> None:
        super().validate_params()

        if not np.isfinite(self.offset):
            raise ValueError(f"{self.offset=} must be finite")
        if not np.isfinite(self.bounds[0]):
            raise ValueError(f"{self.bounds[0]=} must be finite")
        if not (self.bounds[0] <= self.offset <= self.bounds[1]):
            raise ValueError(f"{self.offset=} not in bounds {self.bounds}")

    def encode[Arr: SupportsArrayUfunc](self, data: Arr, /) -> Arr:
        return np.log(data + self.offset)  # type: ignore

    def decode[Arr: SupportsArrayUfunc](self, data: Arr, /) -> Arr:
        return np.maximum(np.exp(data) - self.offset, 0)  # pyrefly: ignore[bad-return]

    def fit(self, data: SupportsArrayUfunc, /) -> None:
        array = np.asanyarray(data)

        if not all((array >= 0) | np.isnan(array)):
            raise ValueError("Data must be in [0, ∞) or NaN.")

        if array.dtype != np.float64:
            warnings.warn(
                "It is not recommended to use this encoder with non-float64 data. "
                f"But {array.dtype=}.",
                RuntimeWarning,
                stacklevel=2,
            )

        match self.METHOD(self.method):
            case self.METHOD.fixed:
                offset = self.offset_guess

            case self.METHOD.minimum:
                offset = array[array > 0].min() / 2

            case self.METHOD.quartile:
                offset = (
                    np.nanquantile(array, 0.25) / np.nanquantile(array, 0.75)
                ) ** 2

            case self.METHOD.match_uniform:
                fun = construct_wasserstein_loss_boxcox_uniform(array)
                x0 = np.float64(self.offset_guess)
                sol = minimize(
                    fun,
                    x0,
                    method="trust-constr",
                    bounds=[self.bounds],
                    options={"disp": self.verbose},
                )
                offset = sol.x

            case self.METHOD.match_normal:
                fun = construct_wasserstein_loss_boxcox_normal(array)
                x0 = np.float64(self.offset_guess)
                sol = minimize(
                    fun,
                    x0,
                    method="trust-constr",
                    bounds=[self.bounds],
                    options={"disp": self.verbose},
                )
                offset = sol.x

            case other:
                assert_never(other)

        self.offset = float(np.asarray(offset).item())


@pretty_repr
@dataclass(slots=True, init=False)
class LogitBoxCoxEncoder[T: SupportsArrayUfunc](FittableEncoder[T, T]):
    r"""Encode data from the unit interval with a fitted offset logit.

    $$ \op{encode}_c(x) = \log\left(\frac{x+c}{1-x+c}\right), \qquad x ∈ [0, 1] $$

    A positive offset $c > 0$ keeps endpoint logarithms finite.
    Decoding clips reconstructions to $[0, 1]$:

    $$ \op{decode}_c(y) = \clip\left(\frac{eʸ+(eʸ-1)c}{1+eʸ}, 0, 1\right) $$

    NaNs are excluded during fitting and propagate through encoding and decoding.
    With $Q₁, Q₃$ denoting the non-NaN data's first and third quartiles:

    1. `fixed`: use `offset_guess` as $c$.
    2. `minimum`: use $c = ¼[\min\set{xᵢ : xᵢ > 0} + \min\set{1-xᵢ : xᵢ < 1}]$.
       At least one observation above zero and one below one are required.
    3. `quartile`: use $c = ½[(Q₁/Q₃)² + ((1-Q₃)/(1-Q₁))²]$, the average of
       the squared quartile ratios for the data and its reflection $1-x$.
       Both denominators must be positive.
    4. `match-uniform` (default) or `match-normal`: minimize the squared Wasserstein-2
       loss of the centered, rescaled data against a uniform or normal target with
       mean zero and variance one. At least two distinct, finite values are required.

    Matching centers and rescales only during fitting; encoding returns the logit.
    """

    class METHOD(StrEnum):
        r"""Methods to fit the Box-Cox parameter."""

        fixed = "fixed"
        minimum = "minimum"
        quartile = "quartile"
        match_normal = "match-normal"
        match_uniform = "match-uniform"

    _: KW_ONLY

    bounds: tuple[float, float] = (0.0, 1.0)
    method: METHOD = METHOD.match_uniform
    offset: float = UNDEFINED
    offset_guess: float = 0.1
    verbose: bool = False

    # FIXME: simplify if a converter option is added to the dataclass.field
    def __init__(
        self,
        *,
        bounds: tuple[float, float] = (0.0, 1.0),
        method: METHOD | str = METHOD.match_uniform,
        offset: float = UNDEFINED,
        offset_guess: float = 0.1,
        verbose: bool = False,
    ) -> None:
        if method not in self.METHOD:
            raise ValueError(f"{method=} unknown. Available: {', '.join(self.METHOD)}")

        self.bounds = bounds
        self.offset_guess = offset_guess
        self.method = self.METHOD(method)
        self.offset = offset
        self.verbose = verbose

        if self.method is self.METHOD.fixed:
            self.offset = self.offset_guess
            self.validate_params()

    def validate_params(self) -> None:
        super().validate_params()
        # FIXME: This appears to be incorrect
        if not (self.bounds[0] <= self.offset <= self.bounds[1]):
            raise ValueError(f"{self.offset=} not in bounds {self.bounds}")

    def encode[Arr: SupportsArrayUfunc](self, data: Arr, /) -> Arr:
        return np.log((data + self.offset) / (1 - (data - self.offset)))  # type: ignore

    def decode[Arr: SupportsArrayUfunc](self, data: Arr, /) -> Arr:
        ey = np.exp(data)
        r = (ey + (ey - 1) * self.offset) / (1 + ey)
        return np.clip(r, 0, 1)  # type: ignore

    def fit(self, data: SupportsArrayUfunc, /) -> None:
        array = np.asanyarray(data)

        if not all(np.isnan(array) | ((array >= 0) & (array <= 1))):
            raise ValueError("Data must be in [0, 1] or NaN.")

        if array.dtype != np.float64:
            warnings.warn(
                "It is not recommended to use this encoder with non-float64 data. "
                f"But {array.dtype=}.",
                RuntimeWarning,
                stacklevel=2,
            )

        match self.method:
            case self.METHOD.fixed:
                offset = self.offset_guess

            case self.METHOD.minimum:
                lower = array[array > 0].min() / 2
                upper = (1 - array[array < 1].max()) / 2
                offset = (lower + upper) / 2

            case self.METHOD.quartile:
                lower = (np.nanquantile(array, 0.25) / np.nanquantile(array, 0.75)) ** 2
                upper = (
                    (1 - np.nanquantile(array, 0.75))
                    / (1 - np.nanquantile(array, 0.25))
                ) ** 2
                offset = (lower + upper) / 2

            case self.METHOD.match_uniform:
                fun = construct_wasserstein_loss_logit_uniform(array)
                x0 = np.float64(self.offset_guess)
                sol = minimize(
                    fun,
                    x0,
                    method="trust-constr",
                    bounds=[self.bounds],
                    options={"disp": False},
                )
                offset = sol.x.squeeze()

            case self.METHOD.match_normal:
                fun = construct_wasserstein_loss_logit_normal(array)
                x0 = np.float64(self.offset_guess)
                sol = minimize(
                    fun,
                    x0,
                    method="trust-constr",
                    bounds=[self.bounds],
                    options={"disp": False},
                )
                offset = sol.x.squeeze()

            case other:
                assert_never(other)

        self.offset = float(np.array(offset).item())
        self.validate_params()


@pretty_repr
@dataclass(slots=True)
class LogEncoder(FittableEncoder[NDArray, NDArray]):
    r"""Encode nonnegative data using base-two logarithms.

    $$ \op{encode}(x) = \begin{cases}\log₂(x)& x>0 \\ \log₂(τ/2)& x≤0\end{cases} $$

    For finite fitting data, $τ$ (`threshold`) is the smallest positive observation;
    at least one positive observation is required. Zeros are represented by
    $\log₂(τ/2)$ (`replacement`), one unit below the smallest observed logarithm.

    Decoding exponentiates and maps reconstructions below the threshold to zero:

    $$ \op{decode}(y) = \begin{cases} 2ʸ & 2ʸ ≥ τ \\ 0 & 2ʸ < τ \end{cases} $$

    Negative fitting values are rejected. NaNs propagate through encoding and
    decoding. They are not excluded during fitting, so a NaN in the fitting data
    also makes the threshold and replacement NaN.
    """

    threshold: NDArray = UNDEFINED
    replacement: NDArray = UNDEFINED

    def fit(self, data: NDArray, /) -> None:
        if np.any(data < 0):
            raise ValueError("Data must be non-negative.")

        mask = data == 0
        self.threshold = data[~mask].min()
        self.replacement = np.log2(self.threshold / 2)

    def encode[Arr: NDArray](self, data: Arr, /) -> Arr:
        result = data.copy()
        mask = data <= 0
        result[:] = np.where(mask, self.replacement, np.log2(data))
        return result

    def decode[Arr: NDArray](self, data: Arr, /) -> Arr:
        result = 2**data
        mask = result < self.threshold
        result[:] = np.where(mask, 0, result)
        return result  # type: ignore


class LogitEncoder[T: SupportsArrayUfunc](StaticEncoder[T, T]):
    r"""Encode probabilities as log odds, without fitting any parameters.

    $$ \op{encode}(x) = \log\left(\frac{x}{1-x}\right), \qquad x ∈ (0, 1) $$

    Every input must lie strictly between zero and one; endpoints and NaNs are
    rejected. Decoding applies the logistic function and clips to $[0, 1]$:

    $$ \op{decode}(y) = \clip\left(\frac{1}{1+e⁻ʸ}, 0, 1\right) $$
    """

    def encode[Arr: SupportsArrayUfunc](self, data: Arr, /) -> Arr:
        # NOTE: do not replace with np.any(data <= 0) since it gives wrong results for NaNs.
        if not np.all((data > 0) & (data < 1)):  # type: ignore
            raise ValueError("Data must be in the range (0, 1).")
        return np.log(data / (1 - data))  # type: ignore

    def decode[Arr: SupportsArrayUfunc](self, data: Arr, /) -> Arr:
        return np.clip(1 / (1 + np.exp(-data)), 0, 1)  # type: ignore
