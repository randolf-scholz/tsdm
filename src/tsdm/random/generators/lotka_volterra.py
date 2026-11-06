r"""Lotka–Volterra Simulation.

References:
    - [Lotka–Volterra equations](https://en.wikipedia.org/wiki/Lotka-Volterra_equations).
"""

__all__ = ["LotkaVolterra"]

from dataclasses import KW_ONLY, dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm as univariate_normal, uniform

from tsdm.random.distributions import RV

from .base import BaseIVP_Generator, Size


@dataclass(slots=True)
class LotkaVolterra(BaseIVP_Generator):
    r"""Lotka–Volterra predator–prey simulation.

    The Lotka–Volterra equations are a pair of first-order nonlinear differential
    equations describing two interacting species, one predator and one prey:

    $$\begin{aligned}
        \dot{x} &= αx - βxy \\
        \dot{y} &= δxy - γy
    \end{aligned}$$

    Here, $x$ is the prey population, $y$ is the predator population, and
    $α$, $β$, $γ$, and $δ$ are positive real parameters describing their interaction.
    """

    X_MIN: ClassVar[float] = 0.0

    _: KW_ONLY

    alpha: float = 1.0
    r"""Prey reproduction rate."""
    beta: float = 0.5
    r"""Prey capture rate."""
    gamma: float = 0.7
    r"""Predator death rate."""
    delta: float = 0.3
    r"""Predator feeding rate."""
    prey0: float = 1.0
    r"""Baseline initial prey population."""
    predator0: float = 1.0
    r"""Baseline initial predator population."""

    @property
    def observation_noise_dist(self) -> RV:
        r"""Noise distribution."""
        return uniform(loc=0.95, scale=0.1)

    @property
    def initial_state_dist(self) -> RV:
        r"""Noise distribution."""
        return univariate_normal(loc=0, scale=1)

    def _get_initial_state_impl(self, *, size: Size = ()) -> NDArray:
        r"""Generate one or more initial states $y₀$."""
        p = self.initial_state_dist
        prey_noise = p.rvs(size=size, random_state=self.rng).clip(-2, +2)
        predator_noise = p.rvs(size=size, random_state=self.rng).clip(-2, +2)
        theta0 = self.prey0 + prey_noise
        omega0 = self.predator0 + predator_noise
        return np.stack([theta0, omega0], axis=-1)

    def _make_observations_impl(self, state: NDArray, /) -> NDArray:
        r"""Create observations from the solution."""
        # multiplicative noise
        p = self.observation_noise_dist
        return state * p.rvs(size=state.shape, random_state=self.rng)

    # Float[..., $N], Float[..., $N, 2] -> Float[..., $N, 2]
    def system(self, t: ArrayLike, state: ArrayLike) -> NDArray:
        r"""Vector field of the Lotka–Volterra model.

        Supported input and output shapes include:

        - `[(...,), (2,)] -> (..., 2)`
        - `[(), (..., 2)] -> (..., 2)`
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
        r"""Project the solution onto the constraint set."""
        if x.min() < (self.X_MIN - tol):
            raise RuntimeError("Integrator produced vastly negative values.")

        return x.clip(min=self.X_MIN)

    def validate_solution(self, x: NDArray, /) -> None:
        r"""Validate constraints on the solution."""
        if (x_min := x.min()) < self.X_MIN:
            raise ValueError(f"Lower bound violated: {x_min}.")
