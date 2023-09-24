"""Lotka–Volterra Simulation.

References:
    - https://en.wikipedia.org/wiki/Lotka-Volterra_equations
"""

__all__ = ["LotkaVolterra"]

from dataclasses import KW_ONLY, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm as univariate_normal, uniform

from tsdm.random.generators._generators import IVP_Generator
from tsdm.random.stats.distributions import Distribution
from tsdm.types.aliases import SizeLike


@dataclass
class LotkaVolterra(IVP_Generator[NDArray]):
    r"""Lotka–Volterra Equations Simulation.

    The Lotka–Volterra equations, also known as the predator–prey equations, are a pair of
    first-order, non-linear, differential equations frequently used to describe the dynamics of
    biological systems in which two species interact, one as a predator and the other as prey.

    .. math::
        \dot{x} = α⋅x - β⋅x⋅y
        \dot{y} = δ⋅x⋅y - γ⋅y

    Here, x is the number of prey, y is the number of some predator, and α, β, γ, δ are positive
    real parameters describing the interaction of the two species.
    """

    _: KW_ONLY
    alpha: float = 1.0
    """Gravitational acceleration."""
    beta: float = 1.0
    """Length of the pendulum."""
    gamma: float = 1.0
    """Mass of the pendulum."""
    delta: float = 1.0
    """Damping coefficient."""
    prey0: float = 1.0
    """Initial angle."""
    predator0: float = 1.0
    """Initial angular velocity."""
    observation_noise: Distribution = uniform(loc=0.95, scale=0.1)  # 5% noise
    """Noise distribution."""
    parameter_noise: Distribution = univariate_normal(loc=0, scale=1)
    """Noise distribution."""

    def get_initial_state(self, size: SizeLike = ()) -> NDArray:
        """Generate (multiple) initial state(s) y₀."""
        theta0 = self.prey0 + self.parameter_noise.rvs(size=size).clip(-2, +2)
        omega0 = self.predator0 + self.parameter_noise.rvs(size=size).clip(-2, +2)
        return np.stack([theta0, omega0], axis=-1)

    def make_observations(self, x: NDArray, /) -> NDArray:
        """Create observations from the solution."""
        # multiplicative noise
        return x * self.observation_noise.rvs(size=x.shape)

    def system(self, t: Any, state: ArrayLike) -> NDArray:
        """Vector field of the pendulum.

        .. Signature:: ``[(...B, N), (...B, N, 2) -> (...B, N, 2)``

        sub-signatures:
            - ``[(...,), (2, ) -> (..., 2)``
            - ``[(,), (..., 2) -> (..., 2)``
        """
        t = np.asarray(t)
        state = np.asarray(state)

        x = state[..., 0]
        y = state[..., 1]
        xy = x * y
        new_state = np.stack(
            [
                self.alpha * x - self.beta * xy,
                self.delta * xy - self.gamma * y,
            ]
        )
        return np.einsum("..., ...d -> ...d", np.ones_like(t), new_state)

    def project_solution(self, x: NDArray, /, *, tol: float = 1e-3) -> NDArray:
        """Project the solution onto the constraint set."""
        assert x.min() > -tol, f"Integrator produced vastly negative values {x.min()}."
        return x.clip(0)

    def validate_solution(self, x: NDArray, /) -> None:
        """Validate constraints on the parameters."""
        assert x.min() >= 0, f"Integrator produced negative values: {x.min()}<0"
