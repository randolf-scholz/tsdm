"""Lotka–Volterra Simulation.

References:
    - https://en.wikipedia.org/wiki/Lotka-Volterra_equations
"""

__all__ = ["LoktaVolterra"]

from dataclasses import KW_ONLY, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm as univariate_normal

from tsdm.random.generators._generators import Distribution, IVP_Generator
from tsdm.types.aliases import SizeLike


@dataclass
class LoktaVolterra(IVP_Generator[np.ndarray]):
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
    observation_noise: Distribution = univariate_normal(loc=0, scale=0.05)
    """Noise distribution."""
    parameter_noise: Distribution = univariate_normal(loc=0, scale=1)
    """Noise distribution."""

    def get_initial_state(self, size: SizeLike = ()) -> np.ndarray:
        """Generate (multiple) initial state(s) y₀."""
        theta0 = self.prey0 + self.parameter_noise.rvs(size=size).clip(-2, +2)
        omega0 = self.predator0 + self.parameter_noise.rvs(size=size).clip(-2, +2)
        return np.stack([theta0, omega0], axis=-1)

    def make_observations(self, sol: Any, /) -> np.ndarray:
        """Create observations from the solution."""
        # add observation noise
        observations = sol.y + self.observation_noise.rvs(size=sol.y.shape)
        return observations

    def system(self, state: ArrayLike, *, t: Any = None) -> np.ndarray:
        """Vector field of the pendulum.

        .. Signature:: ``[(...,), (..., 2) -> (..., 2)``
        """
        if t is not None:
            raise ValueError("Lotka-Volterra equations are a time-invariant system.")

        state = np.asarray(state)
        x = state[..., 0]
        y = state[..., 1]
        xy = x * y

        return np.stack(
            [
                self.alpha * x - self.beta * xy,
                self.delta * xy - self.gamma * y,
            ]
        )
