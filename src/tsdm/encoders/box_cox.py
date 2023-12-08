r"""Box-Cox encoder."""

__all__ = [
    # Constants
    "METHODS",
    # Classes
    "BoxCoxEncoder",
    "LogitBoxCoxEncoder",
]

import warnings
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass

import numpy as np
import pandas as pd
from numpy import pi as PI
from numpy._typing import NDArray
from pandas import Series
from scipy.optimize import minimize
from scipy.special import erfinv
from typing_extensions import ClassVar, Literal, TypeAlias

from tsdm.encoders.base import BaseEncoder
from tsdm.utils.strings import pprint_repr

# region Constants ---------------------------------------------------------------------
ROOT_3 = np.sqrt(3)
METHOD: TypeAlias = Literal[
    None, "minimum", "quartile", "match-normal", "match-uniform"
]
METHODS: list[METHOD] = [None, "minimum", "quartile", "match-normal", "match-uniform"]
# endregion Constants ------------------------------------------------------------------


@pprint_repr
@dataclass
class BoxCoxEncoder(BaseEncoder):
    r"""Encode unbounded non-negative data with a logarithmic transform.

    .. math::
        \text{encode}： ℝ_{≥0} ⟶ ℝ，x ⟼ \log(x+c)
        \text{decode}： ℝ ⟶ ℝ_{≥0}，y ⟼ \max(\exp(y)-c, 0)

    We consider multiple ideas for how to fit the parameter $c$

    1. Half the minimal non-zero value: `c = min(data[data>0])/2`
    2. Square of the first quartile divided by the third quartile (Stahle 2002)
    3. Value which minimizes the Wasserstein distance to
        - a mean-0, variance-1 uniform distribution
        - a mean-0, variance-1 normal distribution
    """

    requires_fit: ClassVar[bool] = True

    _: KW_ONLY

    method: METHOD = "match-uniform"
    initial_value: float = 1.0
    bounds: tuple[float, float] = (0.0, 1.0)
    offset: float = NotImplemented
    verbose: bool = False

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"{self.method=} unknown. Available: {METHODS}")

    @staticmethod
    def construct_loss_wasserstein_uniform(
        x: NDArray, /, *, lower: float = -ROOT_3, upper: float = +ROOT_3
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

        def integrate_quantile(q: NDArray[np.float_]) -> NDArray[np.float_]:
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

        def fun(c: NDArray) -> NDArray:
            u = np.log(np.add.outer(c, unique))
            # transform to target loc-scale
            mean = np.mean(u, axis=-1, keepdims=True)
            stdv = np.std(u, axis=-1, keepdims=True)
            y = (u - mean + μ) * (σ / stdv)
            return np.einsum("...i, i -> ...", y**2 + B * y + C, α)

        return fun

    @staticmethod
    def construct_loss_wasserstein_normal(
        x: NDArray, /, *, loc: float = 0.0, scale: float = 1.0
    ) -> Callable[[NDArray], NDArray]:
        r"""Construct the loss for the Normal distribution.

        .. math::
            W₂² = ∑ₖ [αₖxₖ² -2βₖxₖ + αₖC] = ∑ₖ αₖ[xₖ² -2(βₖ/αₖ)xₖ + C]
            F^{-1}(q) &= μ + σ√2\erf^{-1}(2q-1)
            β &= ∫_a^b F^{-1}(q)dq = (b-a)μ - σ/√(2PI) (e^{-\erf^{-1}(2b-1)^2} - e^{-\erf^{-1}(2a-1)^2}
            C &= ∫_0^1 F^{-1}(q)^2 dq = μ^2 + σ^2
        """

        def integrate_quantile(q: NDArray[np.float_]) -> NDArray[np.float_]:
            if (loc, scale) == (0, 1):
                return -np.exp(-erfinv(2 * q - 1) ** 2) / np.sqrt(2 * PI)
            return loc * q - scale * np.exp(-erfinv(2 * q - 1) ** 2) / np.sqrt(2 * PI)

        μ = loc
        σ = scale
        mask = np.isnan(x)
        unique, counts = np.unique(x[~mask], return_counts=True)
        α = counts / np.sum(counts)
        p = np.insert(np.cumsum(α), 0, 0).clip(0, 1)
        β = integrate_quantile(p[1:]) - integrate_quantile(p[:-1])
        C = 1.0 if (μ, σ) == (0, 1) else μ ** 2 + σ**2
        B = -2 * (β / α)

        def fun(c: NDArray) -> NDArray:
            u = np.log(np.add.outer(c, unique))
            # transform to target loc-scale
            mean = np.mean(u, axis=-1, keepdims=True)
            stdv = np.std(u, axis=-1, keepdims=True)
            y = (u - mean + μ) * (σ / stdv)
            return np.einsum("...i, i -> ...", y**2 + B * y + C, α)

        return fun

    def fit(self, data: Series, /) -> None:
        if not all(pd.isna(data) | (data >= 0)):
            raise ValueError("Data must be in [0, 1] or NaN.")

        if not data.dtype == np.float64:
            warnings.warn(
                "It is not recommended to use this encoder with non-float64 data. "
                f"But {data.dtype=}.",
                RuntimeWarning,
                stacklevel=2,
            )

        match self.method:
            case None:
                offset = self.initial_value
            case "minimum":
                offset = data[data > 0].min() / 2
            case "quartile":
                offset = (np.quantile(data, 0.25) / np.quantile(data, 0.75)) ** 2
            case "match-uniform":
                fun = self.construct_loss_wasserstein_uniform(data)
                x0 = np.array(self.initial_value)
                sol = minimize(
                    fun,
                    x0,
                    method="trust-constr",
                    bounds=[self.bounds],
                    options={"disp": self.verbose},
                )
                offset = sol.x.squeeze()
            case "match-normal":
                fun = self.construct_loss_wasserstein_normal(data)
                x0 = np.array(self.initial_value)
                sol = minimize(
                    fun,
                    x0,
                    method="trust-constr",
                    bounds=[self.bounds],
                    options={"disp": self.verbose},
                )
                offset = sol.x.squeeze()
            case _:
                raise ValueError(f"Unknown method {self.method}")
        self.offset = float(np.array(offset).item())
        assert self.bounds[0] <= self.offset <= self.bounds[1]

    def encode(self, data: Series, /) -> Series:
        # assert all(np.isnan(data) | (data >= 0))
        return np.log(data + self.offset)

    def decode(self, data: Series, /) -> Series:
        return np.maximum(np.exp(data) - self.offset, 0)
        # assert all(np.isnan(decoded) | (decoded >= 0))


@pprint_repr
@dataclass
class LogitBoxCoxEncoder(BaseEncoder):
    r"""Encode data from the interval [0,1] with a logit transform.

    An offset c is added/subtracted to avoid log(0) and division by zero.

    .. math::
        \text{encode}： ℝ_{≥0} ⟶ ℝ，x ⟼ \log((x + c) / (1 - (x - c)))
        \text{decode}： ℝ ⟶ ℝ_{≥0}，y ⟼ \max(\exp(y)-c, 0)

    We consider multiple ideas for how to fit the parameter $c$

    1. Half the minimal non-zero value: `c = min(data[data>0])/2`
    2. Square of the first quartile divided by the third quartile (Stahle 2002)
    3. Value which minimizes the Wasserstein distance to
        - a mean-0, variance-1 uniform distribution
        - a mean-0, variance-1 normal distribution
    """

    requires_fit: ClassVar[bool] = True

    _: KW_ONLY

    method: METHOD = "match-uniform"
    initial_value: float = 0.01
    verbose: bool = False
    offset: float = NotImplemented
    bounds: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"{self.method=} unknown. Available: {METHODS}")

    @staticmethod
    def construct_loss_wasserstein_uniform(
        x: NDArray, /, *, lower: float = -ROOT_3, upper: float = +ROOT_3
    ) -> Callable[[NDArray], NDArray]:
        r"""Construct the loss for the Uniform distribution.

        .. math::
            W₂² = ∑ₖ [αₖxₖ² -2βₖxₖ + αₖC] = ∑ₖ αₖ[xₖ² +Bxₖ + C]
            F^{-1}(q) &= a + (b-a)q
            β &= ∫ F^{-1}(q)dq = aq + ½(b-a)q²
            B &= -2(βₖ/αₖ)
            C &= ∫_0^1 F^{-1}(q)^2 dq = ⅓(a^2 + ab + b^2)

        Also note: (1, 1; -1, 1)(a,b) = (2, 0; 0, √12) (μ, σ)
        Hence: a = μ-√3σ, b = μ+√3σ
        And: μ = ½(a+b), σ² = (a-b)²/12
        """

        def integrate_quantile(q: NDArray[np.float_]) -> NDArray[np.float_]:
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

        def fun(c: NDArray) -> NDArray:
            u = np.log(np.add.outer(c, unique) / (1 + np.add.outer(c, -unique)))
            # transform to target loc-scale
            mean = np.mean(u, axis=-1, keepdims=True)
            stdv = np.std(u, axis=-1, keepdims=True)
            y = (u - mean + μ) * (σ / stdv)
            return np.einsum("...i, i -> ...", y**2 + B * y + C, α)

        return fun

    @staticmethod
    def construct_loss_wasserstein_normal(
        x: NDArray, /, *, loc: float = 0.0, scale: float = 1.0
    ) -> Callable[[NDArray], NDArray]:
        r"""Construct the loss for the Normal distribution.

        .. math::
            W₂² = ∑ₖ [αₖxₖ² -2βₖxₖ + αₖC] = ∑ₖ αₖ[xₖ² -2(βₖ/αₖ)xₖ + C]
            F^{-1}(q) &= μ + σ√2\erf^{-1}(2q-1)
            β &= ∫_a^b F^{-1}(q)dq = (b-a)μ - σ/√(2PI) (e^{-\erf^{-1}(2b-1)^2} - e^{-\erf^{-1}(2a-1)^2}
            C &= ∫_0^1 F^{-1}(q)^2 dq = μ^2 + σ^2
        """

        def integrate_quantile(q: NDArray[np.float_]) -> NDArray[np.float_]:
            if (loc, scale) == (0, 1):
                return -np.exp(-erfinv(2 * q - 1) ** 2) / np.sqrt(2 * PI)
            return loc * q - scale * np.exp(-erfinv(2 * q - 1) ** 2) / np.sqrt(2 * PI)

        μ = loc
        σ = scale
        mask = np.isnan(x)
        unique, counts = np.unique(x[~mask], return_counts=True)
        α = counts / np.sum(counts)
        p = np.insert(np.cumsum(α), 0, 0).clip(0, 1)
        β = integrate_quantile(p[1:]) - integrate_quantile(p[:-1])
        B = -2 * (β / α)
        C = 1.0 if (μ, σ) == (0, 1) else μ ** 2 + σ**2

        def fun(c: NDArray) -> NDArray:
            u = np.log(np.add.outer(c, unique) / (1 + np.add.outer(c, -unique)))
            # transform to target loc-scale
            mean = np.mean(u, axis=-1, keepdims=True)
            stdv = np.std(u, axis=-1, keepdims=True)
            y = (u - mean + μ) * (σ / stdv)
            return np.einsum("...i, i -> ...", y**2 + B * y + C, α)

        return fun

    def fit(self, data: Series, /) -> None:
        if not all(np.isnan(data) | ((data >= 0) & (data <= 1))):
            raise ValueError("Data must be in [0, 1] or NaN.")

        if not data.dtype == np.float64:
            warnings.warn(
                "It is not recommended to use this encoder with non-float64 data. "
                f"But {data.dtype=}.",
                RuntimeWarning,
                stacklevel=2,
            )

        match self.method:
            case None:
                offset = self.initial_value
            case "minimum":
                lower = data[data > 0].min() / 2
                upper = (1 - data[data < 1].max()) / 2
                offset = (lower + upper) / 2
            case "quartile":
                lower = (np.quantile(data, 0.25) / np.quantile(data, 0.75)) ** 2
                upper = (
                    (1 - np.quantile(data, 0.75)) / (1 - np.quantile(data, 0.25))
                ) ** 2
                offset = (lower + upper) / 2
            case "match-uniform":
                fun = self.construct_loss_wasserstein_uniform(data)
                x0 = np.array(self.initial_value)
                sol = minimize(
                    fun,
                    x0,
                    method="trust-constr",
                    bounds=[self.bounds],
                    options={"disp": False},
                )
                offset = sol.x.squeeze()
            case "match-normal":
                fun = self.construct_loss_wasserstein_normal(data)
                x0 = np.array(self.initial_value)
                sol = minimize(
                    fun,
                    x0,
                    method="trust-constr",
                    bounds=[self.bounds],
                    options={"disp": False},
                )
                offset = sol.x.squeeze()
            case _:
                raise ValueError(f"Unknown method {self.method}")

        self.offset = float(np.array(offset).item())
        assert self.bounds[0] <= self.offset <= self.bounds[1]

    def encode(self, data: Series, /) -> Series:
        # assert all(np.isnan(data) | ((data >= 0) & (data <= 1)))
        return np.log(data + self.offset) - np.log((1 - data) + self.offset)

    def decode(self, data: Series, /) -> Series:
        ey = np.exp(data)
        r = (ey + (ey - 1) * self.offset) / (1 + ey)
        return np.clip(r, 0, 1)
        # assert all(np.isnan(decoded) | ((decoded >= 0) & (decoded <= 1)))
