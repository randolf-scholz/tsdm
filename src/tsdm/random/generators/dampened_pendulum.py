"""Dampened Pendulum Simulation.

References:
    - https://en.wikipedia.org/wiki/Pendulum_(mechanics)
"""

__all__ = ["DampedPendulum"]

from dataclasses import KW_ONLY, dataclass
from typing import Any

import numpy as np
import scipy
from numpy.typing import ArrayLike
from scipy.stats import norm as univariate_normal

from tsdm.random.generators._generators import Distribution, IVP_Generator, IVP_Solver
from tsdm.types.aliases import SizeLike


@dataclass
class DampedPendulum(IVP_Generator[np.ndarray]):
    """Dampened Pendulum Simulation.

    The dampended pendulum is an autonomous system with two degrees of freedom.

    References:
        - Neural Continuous-Discrete State Space Models
          Abdul Fatir Ansari, Alvin Heng, Andre Lim, Harold Soh
          Proceedings of the 40th International Conference on Machine Learning
          https://proceedings.mlr.press/v202/ansari23a.html
          https://github.com/clear-nus/NCDSSM
        - Deep Variational Bayes Filters: Unsupervised Learning of State Space Models from Raw Data
          Maximilian Karl, Maximilian Soelch, Justin Bayer, Patrick van der Smagt
          ICLR 2017
          https://openreview.net/forum?id=HyTqHL5xg
        - Deep Rao-Blackwellised Particle Filters for Time Series Forecasting
          Richard Kurle, Syama Sundar Rangapuram, Emmanuel de Bézenac, Stephan Günnemann, Jan Gasthaus
          NeurIPS 2020
          https://proceedings.neurips.cc/paper/2020/hash/afb0b97df87090596ae7c503f60bb23f-Abstract.html
          https://dl.acm.org/doi/10.5555/3495724.3497013
    """

    _: KW_ONLY
    g: float = 9.81
    """Gravitational acceleration."""
    length: float = 1.0
    """Length of the pendulum."""
    mass: float = 1.0
    """Mass of the pendulum."""
    gamma: float = 0.25
    """Damping coefficient."""
    theta0: float = np.pi
    """Initial angle."""
    omega0: float = 4.0
    """Initial angular velocity."""
    ivp_solver: IVP_Solver = scipy.integrate.solve_ivp
    """Solver for the initial value problem."""
    observation_noise: Distribution = univariate_normal(loc=0, scale=0.05)
    """Noise distribution."""
    parameter_noise: Distribution = univariate_normal(loc=0, scale=1)
    """Noise distribution."""

    def get_initial_state(self, size: SizeLike = ()) -> np.ndarray:
        """Generate (multiple) initial state(s) y₀."""
        theta0 = self.theta0 + self.parameter_noise.rvs(size=size).clip(-2, +2)
        omega0 = self.omega0 * self.parameter_noise.rvs(size=size).clip(-2, +2)
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
            raise ValueError("Damped-Pendulum qquations are a time-invariant system.")

        state = np.asarray(state)
        theta = state[..., 0]
        omega = state[..., 1]

        return np.stack(
            [
                omega,
                -(self.g / self.length) * np.sin(theta)
                - self.gamma / self.mass * omega,
            ]
        )
