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

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import truncnorm

from tsdm.random.generators._generators import IVP_GeneratorBase
from tsdm.types.aliases import SizeLike


@dataclass
class BouncingBall(IVP_GeneratorBase[NDArray]):
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

    def _get_initial_state_impl(self, size: SizeLike = ()) -> NDArray:
        """Generate (multiple) initial state(s) y₀."""
        x0 = np.random.uniform(low=self.x_min, high=self.x_max, size=size)
        v0 = np.random.uniform(
            low=self.v_min, high=self.v_max, size=size
        ) * np.random.choice([-1, 1], size=size)
        return np.stack([x0, v0], axis=-1)

    def _make_observations_impl(self, loc: NDArray, /) -> NDArray:
        """Create observations from the solution."""
        x = loc[..., 0]
        # sample from truncated normal distribution
        # cf. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
        lower = (self.x_min - x) / self.y_noise
        upper = (self.x_max - x) / self.y_noise
        y = truncnorm.rvs(lower, upper, loc=x, scale=self.y_noise)
        return y

    def _solve_ivp_impl(self, t: ArrayLike, *, y0: ArrayLike) -> NDArray:
        """Solve the initial value problem.

        NOTE: possibly not properly vectorized.
        """
        # cast to array
        t = np.asarray(t)
        y0 = np.asarray(y0)

        # separate x and v
        x0 = y0[..., 0]
        v0 = y0[..., 1]

        # mod-out period ⇝ at most 2 bounces
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
        v = np.select(
            [
                t <= t1,  # no bounce
                (t > t1) & (t <= t2),  # one bounce
                t > t2,  # two bounces
            ],
            [
                v0,
                -v0,
                v0,
            ],
        )

        # move time axis to the back
        x = np.moveaxis(x, 0, -1)
        v = np.moveaxis(v, 0, -1)

        return np.stack([x, v], axis=-1)

    def validate_solution(self, sol: NDArray, /) -> None:
        """Validate constraints on the parameters."""
        x = sol[..., 0]
        assert x.min() >= -1 and x.max() <= +1, f"{[x.min(), x.max()]} not in [-1,+1]"

    def validate_observations(self, x: NDArray, /) -> None:
        assert x.min() >= -1 and x.max() <= +1, f"{[x.min(), x.max()]} not in [-1,+1]"
