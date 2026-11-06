r"""Implementation of the SIR model from epidemiology."""

__all__ = ["SIR"]

from dataclasses import KW_ONLY, dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tsdm.random.distributions import Dirichlet

from .base import BaseIVP_Generator, Size


@dataclass(slots=True)
class SIR(BaseIVP_Generator):
    r"""SIR model from epidemiology.

    $$\begin{aligned}
        \dot{S} &= -βIS \\
        \dot{I} &= βIS - αI \\
        \dot{R} &= αI \\
        S + I + R &= 1
    \end{aligned}$$

    The compartments are population fractions:

    - $S$: susceptible population.
    - $I$: infected population.
    - $R$: recovered population.

    The parameter $α$ is the recovery rate, and $β$ is the transmission rate.
    For $α > 0$, $𝓡₀ = β/α$ is the basic reproduction number in a fully
    susceptible population. The incidence rate is $-dS/dt = βIS$.
    """

    X_MIN: ClassVar[float] = 0.0
    X_MAX: ClassVar[float] = 1.0

    _: KW_ONLY

    alpha: float = 0.1
    r"""Recovery rate."""
    beta: float = 0.5
    r"""Transmission rate."""

    def _get_initial_state_impl(
        self, *, size: Size = (), weights: ArrayLike = (100, 1, 0)
    ) -> NDArray:
        r"""Generate one or more initial states $y₀$.

        The normalized gamma draws use concentration weights `weights`, defaulting
        to `(100, 1, 0)`. A zero weight fixes that compartment at zero; the default
        therefore has no recovered population. The weights must be nonnegative,
        with at least one positive entry.
        """
        return Dirichlet.rvs(weights, size=size, random_state=self.rng)

    def _make_observations_impl(
        self, sir: NDArray, /, *, noise: float = 0.001
    ) -> NDArray:
        r"""Create observations from the solution.

        Sample normalized gamma draws with concentration weights `[S, I, R] / noise`.
        The noise scale must be positive. Zero population fractions remain zero.
        """
        return Dirichlet.rvs(sir / noise, random_state=self.rng)

    # Float[..., $N], Float[..., $N, 3] -> Float[..., $N, 3]
    def system(self, t: ArrayLike, state: ArrayLike) -> NDArray:
        r"""Vector field of the SIR model."""
        t = np.asarray(t)
        state = np.asarray(state)
        S, I, R = np.moveaxis(state, -1, 0)
        assert S.shape == I.shape == R.shape, "Incompatible shapes."

        x = np.stack(
            [
                -self.beta * I * S,
                self.beta * I * S - self.alpha * I,
                self.alpha * I,
            ]
        )
        return np.einsum("..., ...d -> ...d", np.ones_like(t), x)

    def project_solution(self, x: NDArray, /, *, tol: float = 1e-3) -> NDArray:
        r"""Project the solution onto the constraint set."""
        if x.min() < (self.X_MIN - tol):
            raise RuntimeError("Integrator produced vastly negative values.")
        if x.max() > (self.X_MAX + tol):
            raise RuntimeError("Integrator produced vastly positive values.")
        if not np.allclose(x.sum(axis=-1), 1, atol=tol):
            raise RuntimeError("Constraints vioalted.")

        x = x.clip(min=self.X_MIN, max=self.X_MAX)
        x /= x.sum(axis=-1, keepdims=True)
        return x

    def validate_solution(self, x: NDArray, /) -> None:
        r"""Validate constraints on the solution."""
        if (x_min := x.min()) < self.X_MIN:
            raise ValueError(f"Lower bound violated: {x_min}.")
        if (x_max := x.max()) > self.X_MAX:
            raise ValueError(f"Upper bound violated: {x_max}.")
        if not np.allclose(x.sum(axis=-1), 1):
            raise ValueError("Constraints violated.")
