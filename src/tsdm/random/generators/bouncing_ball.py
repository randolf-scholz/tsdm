"""Bouncing Ball Simulation.

References:
    - https://github.com/rtqichen/torchdiffeq/blob/master/examples/bouncing_ball.py
    - Neural Continuous-Discrete State Space Models
      Abdul Fatir Ansari, Alvin Heng, Andre Lim, Harold Soh
      Proceedings of the 40th International Conference on Machine Learning
      https://proceedings.mlr.press/v202/ansari23a.html
      https://github.com/clear-nus/NCDSSM
"""

__all__ = ["BouncingBall"]

from dataclasses import KW_ONLY, dataclass
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import truncnorm

from tsdm.random.generators._generators import IVP_Generator
from tsdm.types.aliases import SizeLike


@dataclass
class BouncingBall(IVP_Generator[np.ndarray]):
    """Bouncing Ball Simulation.

    NOTE: This simulation differs from the reference in two regards:

    1. The original didn't ensure that x∈[-1,+1].
    2. The original didn't ensure that y∈[-1,+1].

    We fix ① by computing the exact time of the bounce and then
    setting the velocity to the negative of the previous velocity.
    We fix ② by sampling yₖ from a truncated normal distribution.

    For a given velocity v₀, the period is `2/|v₀|`.
    """

    _: KW_ONLY
    x_min: Final[float] = -1.0
    """Lower bound of the ball's position."""
    x_max: Final[float] = +1.0
    """Upper bound of the ball's position."""
    v_min: Final[float] = 0.05
    """Minimum velocity of the ball."""
    v_max: Final[float] = 0.5
    """Maximum velocity of the ball."""
    y_noise: Final[float] = 0.05
    """Standard deviation of the observation noise."""

    def get_initial_state(self, size: SizeLike = ()) -> np.ndarray:
        """Generate (multiple) initial state(s) y₀."""
        x0 = np.random.uniform(low=self.x_min, high=self.x_max, size=size)
        v0 = np.random.uniform(
            low=self.v_min, high=self.v_max, size=size
        ) * np.random.choice([-1, 1], size=size)
        return np.stack([x0, v0], axis=-1)

    def make_observations(self, sol: np.ndarray, /) -> np.ndarray:
        """Create observations from the solution."""
        # sample from truncated normal distribution
        y = truncnorm.rvs(self.x_min, self.x_max, loc=sol, scale=self.y_noise)

        # validate and return
        assert y.min() >= -1 and y.max() <= +1
        return y

    def solve_ivp(self, t: ArrayLike, *, y0: ArrayLike) -> np.ndarray:
        """Solve the initial value problem.

        Signature: ``[(N,), (..., 2)] -> (..., N)``
        """
        # cast to array
        t = np.asarray(t)
        y0 = np.asarray(y0)

        # separate x and v
        x0 = y0[..., 0]
        v0 = y0[..., 1]

        # mod out period ⇝ at most 2 bounces
        half_period = (self.x_max - self.x_min) / abs(v0)
        t = np.mod.outer(t, 2 * half_period)

        next_wall = np.sign(v0)  # the next wall the ball will hit
        t1 = (next_wall - x0) / v0  # first bounce
        t2 = t1 + half_period  # second bounce

        # 3-fold distinction
        x = np.select(
            [
                t <= t1,  # no bounce
                (t > t1) & (t <= t2),  # one bounce
                t > t2,  # two bounces
            ],
            [
                x0 + v0 * t,
                next_wall - v0 * (t - t1),
                -next_wall + v0 * (t - t2),
            ],
        )

        # move time axis to the back
        x = np.moveaxis(x, 0, -1)

        # validate and return
        assert x.min() >= -1 and x.max() <= +1
        return x


def example():
    """Example usage of the bouncing ball generator."""
    t = np.linspace(-10, 20, 1000)
    y0 = np.random.uniform(-0.5, 0.7, size=(5, 2))
    x = BouncingBall().solve_ivp(t, y0=y0)
    plt.plot(t, x[0], t, x[1], t, x[3])
