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
from typing import assert_never

import numpy as np
from numpy import pi as PI
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import erfinv

from tsdm.constants import FLOAT, UNDEFINED
from tsdm.pprint import pprint_repr
from tsdm.types import SupportsArrayUfunc

from .base import FittableEncoder, StaticEncoder


def construct_wasserstein_loss_boxcox_uniform(
    x: NDArray,
    /,
    *,
    lower: float = -FLOAT.ROOT_3,
    upper: float = +FLOAT.ROOT_3,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    r"""Construct the loss for the Uniform distribution.

    Given an empirical distribution $x$, we assume $x$ is transformed by Box-Cox transform
    $y(c) = \log(x+c)$ and subsequently normalized $z(c) = (y - μ) / σ$,
    where $μ$ and $σ$ are the mean and standard deviation of $y$.

    Returns:
        Callable[[NDArray], NDArray]: The loss function that maps c to the squared
            Wasserstein-2 loss between $z(c)$ and Uniform distribution with given mean and std.

    .. math::
        W₂² &= ∑ₖ [αₖxₖ² -2βₖxₖ + αₖC] = ∑ₖ αₖ[xₖ² -2(βₖ/αₖ)xₖ + C]  \\
        F⁻¹(q) &= a + (b-a)q                                         \\
        β &= ∫ F⁻¹(q)dq = aq + ½(b-a)q²                              \\
        C &= ∫₀¹ F⁻¹(q)² dq = ⅓(a² + ab + b²)

    Also note: $\bmat{1&1\\-1&1}⋅\bmat{a\\b} = \bmat{2&0\\0&\sqrt{12}}⋅\bmat{μ\\σ}$
    Hence: $\bmat{a\\b}=\bmat{μ-√3σ\\μ+√3σ}$ and $\bmat{μ\\σ}=\bmat{½(a+b)\\(a-b)/\sqrt{12}}$.
    """
    ROOT_3 = FLOAT.ROOT_3

    def integrate_quantile(q: NDArray[np.float64]) -> NDArray[np.float64]:
        if (lower, upper) == (-ROOT_3, +ROOT_3):
            return ROOT_3 * q * (q - 1)
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
    C = 1.0 if (a, b) == (-ROOT_3, +ROOT_3) else (a**2 + a * b + b**2) / 3

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
    r"""Constructs the Wasserstein-2 loss.

    Given an empirical distribution $x$, we assume $x$ is transformed by Box-Cox transform
    $y(c) = \log(x+c)$ and subsequently normalized $z(c) = (y - μ) / σ$,
    where $μ$ and $σ$ are the mean and standard deviation of $y$.

    Returns:
        Callable[[NDArray], NDArray]: The loss function that maps c to the squared
            Wasserstein-2 loss between $z(c)$ and $N(μ,σ²)$.

    .. math::
        W₂² &= ∑ₖ [αₖxₖ² -2βₖxₖ + αₖC] = ∑ₖ αₖ[xₖ² -2(βₖ/αₖ)xₖ + C]      \\
        F⁻¹(q) &= μ + σ√2\erf⁻¹(2q-1)                                    \\
        β &= ∫ₐᵇ F⁻¹(q)dq                                                \\
          &= (b-a)μ - σ/√(2PI) (e^{-\erf⁻¹(2b-1)²} - e^{-\erf⁻¹(2a-1)²}  \\
        C &= ∫₀¹ F⁻¹(q)² dq = μ² + σ²
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
    lower: float = -FLOAT.ROOT_3,
    upper: float = +FLOAT.ROOT_3,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    r"""Construct the loss for the Uniform distribution.

    Given an empirical distribution $x$, we assume $x$ is transformed by logit transofrm
    $y(c) = \log((x+c)/(1-x+c))$ and subsequently normalized: $z(c) = (y - μ) / σ$,
    where $μ$ and $σ$ are the mean and standard deviation of $y$.

    Returns:
        Callable[[NDArray], NDArray]: The loss function that maps c to the squared
            Wasserstein-2 loss between $z(c)$ and $N(μ,σ²)$.

    .. math::
        W₂² &= ∑ₖ [αₖxₖ² -2βₖxₖ + αₖC] = ∑ₖ αₖ[xₖ² +Bxₖ + C]  \\
        F⁻¹(q) &= a + (b-a)q                                  \\
        β &= ∫ F⁻¹(q)dq = aq + ½(b-a)q²                       \\
        B &= -2(βₖ/αₖ)                                        \\
        C &= ∫₀¹ F⁻¹(q)² dq = ⅓(a² + ab + b²)

    Also note: $\bmat{1&1\\-1&1}⋅\bmat{a\\b} = \bmat{2&0\\0&\sqrt{12}}⋅\bmat{μ\\σ}$
    Hence: $\bmat{a\\b}=\bmat{μ-√3σ\\μ+√3σ}$ and $\bmat{μ\\σ}=\bmat{½(a+b)\\(a-b)/\sqrt{12}}$.
    """
    ROOT_3 = FLOAT.ROOT_3

    def integrate_quantile(q: NDArray[np.float64]) -> NDArray[np.float64]:
        if (lower, upper) == (-ROOT_3, +ROOT_3):
            return ROOT_3 * q * (q - 1)
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
    C = 1.0 if (a, b) == (-ROOT_3, +ROOT_3) else (a**2 + a * b + b**2) / 3

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
    r"""Construct the loss for the Normal distribution.

    Given an empirical distribution $x$, we assume $x$ is transformed by logit transofrm
    $y(c) = \log((x+c)/(1-x+c))$ and subsequently normalized: $z(c) = (y - μ) / σ$,
    where $μ$ and $σ$ are the mean and standard deviation of $y$.

    Returns:
        Callable[[NDArray], NDArray]: The loss function that maps c to the squared
            Wasserstein-2 loss between $z(c)$ and $N(μ,σ²)$.

    .. math::
        W₂² &= ∑ₖ [αₖxₖ² -2βₖxₖ + αₖC] = ∑ₖ αₖ[xₖ² -2(βₖ/αₖ)xₖ + C]      \\
        F⁻¹(q) &= μ + σ√2\erf⁻¹(2q-1)                                    \\
        β &= ∫ₐᵇ F⁻¹(q)dq                                                \\
          &= (b-a)μ - σ/√(2PI) (e^{-\erf⁻¹(2b-1)²} - e^{-\erf⁻¹(2a-1)²}  \\
        C &= ∫₀¹ F⁻¹(q)² dq = μ² + σ²
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


@pprint_repr
@dataclass(slots=True, init=False)
class BoxCoxEncoder[T: SupportsArrayUfunc](FittableEncoder[T, T]):
    r"""Encode unbounded non-negative data with a logarithmic transform.

    .. math::
        \text{encode}&： ℝ_{≥0} ⟶ ℝ，x ⟼ \log(x+c)            \\
        \text{decode}&： ℝ ⟶ ℝ_{≥0}，y ⟼ \max(eʸ-c, 0)

    We consider multiple ideas for how to fit the parameter $c$

    1. Half the minimal non-zero value: `c = min(data[data>0])/2`
    2. Square of the first quartile divided by the third quartile (Stahle 2002)
    3. Value which minimizes the Wasserstein distance to
        - a mean-0, variance-1 uniform distribution
        - a mean-0, variance-1 normal distribution
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
        return np.maximum(np.exp(data) - self.offset, 0)  # type: ignore

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


@pprint_repr
@dataclass(slots=True, init=False)
class LogitBoxCoxEncoder[T: SupportsArrayUfunc](FittableEncoder[T, T]):
    r"""Encode data from the interval [0,1] with a logit transform.

    An offset c is added/subtracted to avoid log(0) and division by zero.

    .. math::
        \text{encode}&： ℝ_{≥0} ⟶ ℝ，x ⟼ \log((x + c) / (1 - (x - c)))  \\
        \text{decode}&： ℝ ⟶ ℝ_{≥0}，y ⟼ \max(eʸ-c, 0)

    We consider multiple ideas for how to fit the parameter $c$

    1. Half the minimal non-zero value: `c = min(data[data>0])/2`
    2. Square of the first quartile divided by the third quartile (Stahle 2002)
    3. Value which minimizes the Wasserstein distance to
        - a mean-0, variance-1 uniform distribution
        - a mean-0, variance-1 normal distribution
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


@pprint_repr
@dataclass(slots=True)
class LogEncoder(FittableEncoder[NDArray, NDArray]):
    r"""Encode data on a logarithmic scale.

    Values below the threshold are replaced with a constant value to avoid -inf and NaN values.

    Uses base 2 by default for lower numerical error and fast computation.
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
    r"""Logit encoder."""

    def encode[Arr: SupportsArrayUfunc](self, data: Arr, /) -> Arr:
        # NOTE: do not replace with np.any(data <= 0) since it gives wrong results for NaNs.
        if not np.all((data > 0) & (data < 1)):  # type: ignore
            raise ValueError("Data must be in the range (0, 1).")
        return np.log(data / (1 - data))  # type: ignore

    def decode[Arr: SupportsArrayUfunc](self, data: Arr, /) -> Arr:
        return np.clip(1 / (1 + np.exp(-data)), 0, 1)  # type: ignore
