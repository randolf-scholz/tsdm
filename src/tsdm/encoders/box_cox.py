r"""Box-Cox encoder."""

__all__ = [
    # Classes
    "BoxCoxEncoder",
    "LogitBoxCoxEncoder",
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
from typing import Literal, assert_never

import numpy as np
import pandas as pd
from numpy import pi as PI
from numpy.typing import NDArray
from pandas import Series
from scipy.optimize import minimize
from scipy.special import erfinv

from tsdm.constants import ROOT_3
from tsdm.encoders.base import BaseEncoder
from tsdm.utils.decorators import pprint_repr

# region Constants ---------------------------------------------------------------------


# endregion Constants ------------------------------------------------------------------


def construct_wasserstein_loss_boxcox_uniform(
    x: NDArray, /, *, lower: float = -ROOT_3, upper: float = +ROOT_3
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    r"""Construct the loss for the Uniform distribution.

    Given an empirical distribution $x$, we assume $x$ is transformed by Box-Cox transform
    $y(c) = \log(x+c)$ and subsequently normalized $z(c) = (y - μ) / σ$,
    where $μ$ and $σ$ are the mean and standard deviation of $y$.

    Returns:
        Callable[[NDArray], NDArray]: The loss function that maps c to the squared
            Wasserstein-2 loss between $z(c)$ and Uniform distribution with given mean and std.

    .. math::
        W₂² &= ∑ₖ [αₖxₖ² -2βₖxₖ + αₖC] = ∑ₖ αₖ[xₖ² -2(βₖ/αₖ)xₖ + C]       \\
        F^{-1}(q) &= a + (b-a)q                                           \\
        β &= ∫ F^{-1}(q)dq = aq + ½(b-a)q²                                \\
        C &= ∫_0^1 F^{-1}(q)^2 dq = ⅓(a^2 + ab + b^2)

    Also note: $\bmat{1&1\\-1&1}⋅\bmat{a\\b} = \bmat{2&0\\0&√12}⋅\bmat{μ\\σ}$
    Hence: $\bmat{a\\b}=\bmat{μ-√3σ\\μ+√3σ}$ and $\bmat{μ\\σ}=\bmat{½(a+b)\\(a-b)/√12}$.
    """

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
        W₂² &= ∑ₖ [αₖxₖ² -2βₖxₖ + αₖC] = ∑ₖ αₖ[xₖ² -2(βₖ/αₖ)xₖ + C]                           \\
        F^{-1}(q) &= μ + σ√2\erf^{-1}(2q-1)                                                   \\
        β &= ∫_a^b F^{-1}(q)dq                                                                \\
          &= (b-a)μ - σ/√(2PI) (e^{-\erf^{-1}(2b-1)^2} - e^{-\erf^{-1}(2a-1)^2}               \\
        C &= ∫_0^1 F^{-1}(q)^2 dq = μ^2 + σ^2
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
    x: NDArray, /, *, lower: float = -ROOT_3, upper: float = +ROOT_3
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    r"""Construct the loss for the Uniform distribution.

    Given an empirical distribution $x$, we assume $x$ is transformed by logit transofrm
    $y(c) = \log((x+c)/(1-x+c))$ and subsequently normalized: $z(c) = (y - μ) / σ$,
    where $μ$ and $σ$ are the mean and standard deviation of $y$.

    Returns:
        Callable[[NDArray], NDArray]: The loss function that maps c to the squared
            Wasserstein-2 loss between $z(c)$ and $N(μ,σ²)$.

    .. math::
        W₂² &= ∑ₖ [αₖxₖ² -2βₖxₖ + αₖC] = ∑ₖ αₖ[xₖ² +Bxₖ + C]                      \\
        F^{-1}(q) &= a + (b-a)q                                                   \\
        β &= ∫ F^{-1}(q)dq = aq + ½(b-a)q²                                        \\
        B &= -2(βₖ/αₖ)                                                            \\
        C &= ∫_0^1 F^{-1}(q)^2 dq = ⅓(a^2 + ab + b^2)

    Also note: $\bmat{1&1\\-1&1}⋅\bmat{a\\b} = \bmat{2&0\\0&√12}⋅\bmat{μ\\σ}$
    Hence: $\bmat{a\\b}=\bmat{μ-√3σ\\μ+√3σ}$ and $\bmat{μ\\σ}=\bmat{½(a+b)\\(a-b)/√12}$.
    """

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
        W₂² &= ∑ₖ [αₖxₖ² -2βₖxₖ + αₖC] = ∑ₖ αₖ[xₖ² -2(βₖ/αₖ)xₖ + C]               \\
        F^{-1}(q) &= μ + σ√2\erf^{-1}(2q-1)                                       \\
        β &= ∫_a^b F^{-1}(q)dq                                                    \\
          &= (b-a)μ - σ/√(2PI) (e^{-\erf^{-1}(2b-1)^2} - e^{-\erf^{-1}(2a-1)^2}   \\
        C &= ∫_0^1 F^{-1}(q)^2 dq = μ^2 + σ^2
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
@dataclass
class BoxCoxEncoder(BaseEncoder):
    r"""Encode unbounded non-negative data with a logarithmic transform.

    .. math::
        \text{encode}&： ℝ_{≥0} ⟶ ℝ，x ⟼ \log(x+c)                            \\
        \text{decode}&： ℝ ⟶ ℝ_{≥0}，y ⟼ \max(\exp(y)-c, 0)

    We consider multiple ideas for how to fit the parameter $c$

    1. Half the minimal non-zero value: `c = min(data[data>0])/2`
    2. Square of the first quartile divided by the third quartile (Stahle 2002)
    3. Value which minimizes the Wasserstein distance to
        - a mean-0, variance-1 uniform distribution
        - a mean-0, variance-1 normal distribution
    """

    class METHODS(StrEnum):
        r"""Methods to fit the Box-Cox parameter."""

        fixed = "fixed"
        minimum = "minimum"
        quartile = "quartile"
        match_normal = "match-normal"
        match_uniform = "match-uniform"

    type METHOD = Literal[
        "fixed", "minimum", "quartile", "match-normal", "match-uniform"
    ]

    _: KW_ONLY

    method: METHOD = "match-uniform"
    initial_value: float = 1.0
    bounds: tuple[float, float] = (0.0, 1.0)
    offset: float = NotImplemented
    verbose: bool = False

    def __post_init__(self):
        if self.method not in self.METHODS:
            raise ValueError(f"{self.method=} unknown. Available: {self.METHODS}")
        if self.method == self.METHODS.fixed:
            self.offset = self.initial_value
            self.check_bounds()

    def check_bounds(self) -> None:
        if not (self.bounds[0] <= self.offset <= self.bounds[1]):
            raise ValueError(f"{self.offset=} not in bounds {self.bounds}")

    def encode[T: (NDArray, Series)](self, data: T, /) -> T:
        return np.log(data + self.offset)

    def decode[T: (NDArray, Series)](self, data: T, /) -> T:
        return np.maximum(np.exp(data) - self.offset, 0)

    def fit(self, data: NDArray | Series, /) -> None:
        if not all(pd.isna(data) | (data >= 0)):
            raise ValueError("Data must be in [0, 1] or NaN.")

        if data.dtype != np.float64:
            warnings.warn(
                "It is not recommended to use this encoder with non-float64 data. "
                f"But {data.dtype=}.",
                RuntimeWarning,
                stacklevel=2,
            )

        match self.METHODS(self.method):
            case self.METHODS.fixed:
                offset = self.initial_value
            case self.METHODS.minimum:
                offset = data[data > 0].min() / 2
            case self.METHODS.quartile:
                offset = (np.quantile(data, 0.25) / np.quantile(data, 0.75)) ** 2
            case self.METHODS.match_uniform:
                fun = construct_wasserstein_loss_boxcox_uniform(data)
                x0 = np.array(self.initial_value)
                sol = minimize(
                    fun,
                    x0,
                    method="trust-constr",
                    bounds=[self.bounds],
                    options={"disp": self.verbose},
                )
                offset = sol.x.squeeze()
            case self.METHODS.match_normal:
                fun = construct_wasserstein_loss_boxcox_normal(data)
                x0 = np.array(self.initial_value)
                sol = minimize(
                    fun,
                    x0,
                    method="trust-constr",
                    bounds=[self.bounds],
                    options={"disp": self.verbose},
                )
                offset = sol.x.squeeze()
            case other:
                assert_never(other)

        self.offset = float(np.array(offset).item())
        self.check_bounds()


@pprint_repr
@dataclass
class LogitBoxCoxEncoder(BaseEncoder):
    r"""Encode data from the interval [0,1] with a logit transform.

    An offset c is added/subtracted to avoid log(0) and division by zero.

    .. math::
        \text{encode}&： ℝ_{≥0} ⟶ ℝ，x ⟼ \log((x + c) / (1 - (x - c)))       \\
        \text{decode}&： ℝ ⟶ ℝ_{≥0}，y ⟼ \max(\exp(y)-c, 0)

    We consider multiple ideas for how to fit the parameter $c$

    1. Half the minimal non-zero value: `c = min(data[data>0])/2`
    2. Square of the first quartile divided by the third quartile (Stahle 2002)
    3. Value which minimizes the Wasserstein distance to
        - a mean-0, variance-1 uniform distribution
        - a mean-0, variance-1 normal distribution
    """

    class METHODS(StrEnum):
        r"""Methods to fit the Box-Cox parameter."""

        fixed = "fixed"
        minimum = "minimum"
        quartile = "quartile"
        match_normal = "match-normal"
        match_uniform = "match-uniform"

    type METHOD = Literal[
        "fixed", "minimum", "quartile", "match-normal", "match-uniform"
    ]

    _: KW_ONLY

    method: METHOD = "match-uniform"
    initial_value: float = 0.01
    verbose: bool = False
    offset: float = NotImplemented
    bounds: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        if self.method not in self.METHODS:
            raise ValueError(f"{self.method=} unknown. Available: {self.METHODS}")
        if self.method == self.METHODS.fixed:
            self.offset = self.initial_value
            self.check_bounds()

    def check_bounds(self) -> None:
        if not (self.bounds[0] <= self.offset <= self.bounds[1]):
            raise ValueError(f"{self.offset=} not in bounds {self.bounds}")

    def encode[T: (NDArray, Series)](self, data: T, /) -> T:
        return np.log(data + self.offset) - np.log((1 - data) + self.offset)

    def decode[T: (NDArray, Series)](self, data: T, /) -> T:
        ey = np.exp(data)
        r = (ey + (ey - 1) * self.offset) / (1 + ey)
        return np.clip(r, 0, 1)

    def fit(self, data: NDArray | Series, /) -> None:
        if not all(np.isnan(data) | ((data >= 0) & (data <= 1))):
            raise ValueError("Data must be in [0, 1] or NaN.")

        if data.dtype != np.float64:
            warnings.warn(
                "It is not recommended to use this encoder with non-float64 data. "
                f"But {data.dtype=}.",
                RuntimeWarning,
                stacklevel=2,
            )

        match self.METHODS(self.method):
            case self.METHODS.fixed:
                offset = self.initial_value
            case self.METHODS.minimum:
                lower = data[data > 0].min() / 2
                upper = (1 - data[data < 1].max()) / 2
                offset = (lower + upper) / 2
            case self.METHODS.quartile:
                lower = (np.quantile(data, 0.25) / np.quantile(data, 0.75)) ** 2
                upper = (
                    (1 - np.quantile(data, 0.75)) / (1 - np.quantile(data, 0.25))
                ) ** 2
                offset = (lower + upper) / 2
            case self.METHODS.match_uniform:
                fun = construct_wasserstein_loss_logit_uniform(data)
                x0 = np.array(self.initial_value)
                sol = minimize(
                    fun,
                    x0,
                    method="trust-constr",
                    bounds=[self.bounds],
                    options={"disp": False},
                )
                offset = sol.x.squeeze()
            case self.METHODS.match_normal:
                fun = construct_wasserstein_loss_logit_normal(data)
                x0 = np.array(self.initial_value)
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
        self.check_bounds()
