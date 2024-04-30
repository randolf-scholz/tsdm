"""Lotka–Volterra Simulation.

References:
    - https://en.wikipedia.org/wiki/Lotka-Volterra_equations
"""

__all__ = ["LotkaVolterra"]

from dataclasses import KW_ONLY, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm as univariate_normal, uniform
from typing_extensions import ClassVar

from tsdm.random.distributions import RV
from tsdm.random.generators.base import IVP_GeneratorBase
from tsdm.types.aliases import Size


@dataclass
class LotkaVolterra(IVP_GeneratorBase[NDArray]):
    r"""Lotka-Volterra Equations Simulation.

    The Lotka–Volterra equations, also known as the predator–prey equations, are a pair of
    first-order, non-linear, differential equations frequently used to describe the dynamics of
    biological systems in which two species interact, one as a predator and the other as prey.

    .. math::
        \dot{x} = α⋅x - β⋅x⋅y
        \dot{y} = δ⋅x⋅y - γ⋅y

    Here, x is the number of prey, y is the number of predators, and α, β, γ, δ are positive
    real parameters describing the interaction of the two species.
    """

    X_MIN: ClassVar[float] = 0.0

    _: KW_ONLY

    alpha: float = 1.0
    r"""Prey reproduciton rate."""
    beta: float = 0.5
    r"""Prey capture rate."""
    gamma: float = 0.7
    r"""Predator death rate."""
    delta: float = 0.3
    r"""Predator feeding rate."""
    prey0: float = 1.0
    r"""Initial angle."""
    predator0: float = 1.0
    r"""Initial angular velocity."""

    @property
    def observation_noise_dist(self) -> RV:
        r"""Noise distribution."""
        return uniform(loc=0.95, scale=0.1)

    @property
    def initial_state_dist(self) -> RV:
        r"""Noise distribution."""
        return univariate_normal(loc=0, scale=1)

    def _get_initial_state_impl(self, *, size: Size = ()) -> NDArray:
        r"""Generate (multiple) initial state(s) y₀."""
        p = self.initial_state_dist
        prey_noise = p.rvs(size=size, random_state=self.rng).clip(-2, +2)
        predator_noise = p.rvs(size=size, random_state=self.rng).clip(-2, +2)
        theta0 = self.prey0 + prey_noise
        omega0 = self.predator0 + predator_noise
        return np.stack([theta0, omega0], axis=-1)

    def _make_observations_impl(self, x: NDArray, /) -> NDArray:
        r"""Create observations from the solution."""
        # multiplicative noise
        p = self.observation_noise_dist
        return x * p.rvs(size=x.shape, random_state=self.rng)

    def system(self, t: ArrayLike, state: ArrayLike) -> NDArray:
        r"""Vector field of the pendulum.

        .. signature:: ``[(...B, N), (...B, N, 2) -> (...B, N, 2)``

        sub-signatures:
            - ``[(...,), (2, ) -> (..., 2)``
            - ``[(,), (..., 2) -> (..., 2)``
        """
        t = np.asarray(t)
        state = np.asarray(state)

        x = state[..., 0]
        y = state[..., 1]
        xy = x * y
        new_state = np.stack([
            self.alpha * x - self.beta * xy,
            self.delta * xy - self.gamma * y,
        ])
        return np.einsum("..., ...d -> ...d", np.ones_like(t), new_state)

    def project_solution(self, x: NDArray, /, *, tol: float = 1e-3) -> NDArray:
        r"""Project the solution onto the constraint set."""
        if x.min() < (self.X_MIN - tol):
            raise RuntimeError("Integrator produced vastly negative values.")

        return x.clip(min=self.X_MIN)

    def validate_solution(self, x: NDArray, /) -> None:
        r"""Validate constraints on the parameters."""
        if (x_min := x.min()) < self.X_MIN:
            raise ValueError(f"Lower bound violated: {x_min}.")
