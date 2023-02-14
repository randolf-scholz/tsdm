r"""Box-Cox encoder."""

__all__ = [
    # Classes
    "BoxCoxEncoder",
    "LogitBoxCoxEncoder",
]

import warnings
from typing import Callable, Literal, Optional, TypeAlias

import numpy as np
import pandas as pd
from numpy import pi as PI
from numpy._typing import NDArray
from pandas import Series
from scipy.optimize import minimize
from scipy.special import erfinv

from tsdm.encoders.base import BaseEncoder
from tsdm.utils.strings import repr_mapping

METHOD: TypeAlias = Literal[
    None, "minimum", "quartile", "match-normal", "match-uniform"
]

ROOT_3 = np.sqrt(3)


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

    AVAILABLE_METHODS = [None, "minimum", "quartile", "match-normal", "match-uniform"]

    method: METHOD = "match-uniform"
    offset: float
    initial_params: Optional[np.ndarray] = None
    verbose: bool = False

    def __init__(
        self,
        *,
        method: METHOD = "match-uniform",
        initial_param: Optional[np.ndarray] = None,
    ) -> None:
        if method not in self.AVAILABLE_METHODS:
            raise ValueError(f"{method=} unknown. Available: {self.AVAILABLE_METHODS}")
        if method is None and initial_param is None:
            raise ValueError("Needs to provide initial param if no fitting.")

        self.method = method
        self.initial_param = initial_param
        super().__init__()

    def __repr__(self) -> str:
        return repr_mapping(
            {"method": self.method, "offset": self.offset},
            title=self.__class__.__name__,
            identifier="Encoder",
        )

    @staticmethod
    def construct_loss_wasserstein_uniform(
        x: NDArray, a: float = -ROOT_3, b: float = +ROOT_3
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

            def integrate_quantile(q: NDArray[np.float_]) -> NDArray[np.float_]:
                return np.sqrt(3) * q * (q - 1)

        else:
            C = (a**2 + a * b + b**2) / 3

            def integrate_quantile(q: NDArray[np.float_]) -> NDArray[np.float_]:
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

            def integrate_quantile(q: NDArray[np.float_]) -> NDArray[np.float_]:
                return -np.exp(-erfinv(2 * q - 1) ** 2) / np.sqrt(2 * PI)

        else:
            C = μ**2 + σ**2

            def integrate_quantile(q: NDArray[np.float_]) -> NDArray[np.float_]:
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
                assert self.initial_param is not None
                offset = self.initial_param
            case "minimum":
                offset = data[data > 0].min() / 2
            case "quartile":
                offset = (np.quantile(data, 0.25) / np.quantile(data, 0.75)) ** 2
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
                    options={"disp": self.verbose},
                )
                offset = sol.x.squeeze()
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
                    options={"disp": self.verbose},
                )
                offset = sol.x.squeeze()
            case _:
                raise ValueError(f"Unknown method {self.method}")
        self.offset = float(np.array(offset).item())

    def encode(self, data: Series, /) -> Series:
        # assert all(np.isnan(data) | (data >= 0)), f"{data=}"
        return np.log(data + self.offset)

    def decode(self, data: Series, /) -> Series:
        return np.maximum(np.exp(data) - self.offset, 0)


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

    AVAILABLE_METHODS = [None, "minimum", "quartile", "match-normal", "match-uniform"]

    method: METHOD = "match-normal"
    offset: float

    initial_param: Optional[np.ndarray] = None
    verbose: bool = False

    def __init__(
        self,
        *,
        method: METHOD = "match-normal",
        initial_param: Optional[np.ndarray] = None,
    ) -> None:
        if method not in self.AVAILABLE_METHODS:
            raise ValueError(f"{method=} unknown. Available: {self.AVAILABLE_METHODS}")
        if method is None and initial_param is None:
            raise ValueError("Needs to provide initial param if no fitting.")

        self.method = method
        self.initial_param = initial_param
        super().__init__()

    def __repr__(self) -> str:
        return repr_mapping(
            {"method": self.method, "offset": self.offset},
            title=self.__class__.__name__,
            identifier="Encoder",
        )

    @staticmethod
    def construct_loss_wasserstein_uniform(
        x: NDArray, a: float = -ROOT_3, b: float = +ROOT_3
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

            def integrate_quantile(q: NDArray[np.float_]) -> NDArray[np.float_]:
                return np.sqrt(3) * q * (q - 1)

        else:
            C = (a**2 + a * b + b**2) / 3

            def integrate_quantile(q: NDArray[np.float_]) -> NDArray[np.float_]:
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

            def integrate_quantile(q: NDArray[np.float_]) -> NDArray[np.float_]:
                return -np.exp(-erfinv(2 * q - 1) ** 2) / np.sqrt(2 * PI)

        else:
            C = μ**2 + σ**2

            def integrate_quantile(q: NDArray[np.float_]) -> NDArray[np.float_]:
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
                assert self.initial_param is not None
                offset = self.initial_param
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
                x0 = np.array([1.0])
                sol = minimize(
                    fun,
                    x0,
                    method="trust-constr",
                    # jac=jac,
                    # hess=hess,
                    bounds=[(0, np.inf)],
                    options={"disp": False},
                )
                offset = sol.x.squeeze()
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
                    options={"disp": False},
                )
                offset = sol.x.squeeze()
            case _:
                raise ValueError(f"Unknown method {self.method}")
        self.offset = float(np.array(offset).item())

    def encode(self, data: Series, /) -> Series:
        # assert all(np.isnan(data) | ((data >= 0) & (data <= 1))), f"{data=}"
        return np.log((data + self.offset) / (1 - data + self.offset))

    def decode(self, data: Series, /) -> Series:
        ey = np.exp(-data)
        r = (1 + (1 - ey) * self.offset) / (1 + ey)
        return np.clip(r, 0, 1)
