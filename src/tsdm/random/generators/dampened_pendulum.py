r"""Damped pendulum simulation.

References:
    - [Pendulum mechanics](https://en.wikipedia.org/wiki/Pendulum_(mechanics)).
"""

__all__ = ["DampedPendulum", "DampedPendulumXY"]

from dataclasses import KW_ONLY, dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm as univariate_normal, truncnorm

from tsdm.random.distributions import RV

from .base import BaseIVP_Generator, Size


@dataclass(slots=True)
class DampedPendulum(BaseIVP_Generator):
    r"""Damped pendulum simulation.

    The pendulum has one mechanical degree of freedom, represented by the angle
    $θ$ and angular velocity $ω$:

    $$\begin{aligned}
    dθ/dt &= ω \\
    dω/dt &= -(g/l)\sin θ - (γ/m)ω
    \end{aligned}$$

    Note:
        At the upper equilibrium $θ=π$, $ω=0$, the Jacobian is

        $$ J = \begin{pmatrix}0 & 1 \\ g/l & -γ/m\end{pmatrix} $$

        Its eigenvalues are

        $$ λ = \frac{1}{2}\left(-γ/m ± \sqrt{γ²/m² + 4g/l}\right) $$

        For positive $g$, $l$, and $m$ and nonnegative damping $γ$, one eigenvalue
        is positive. The upper equilibrium is therefore unstable.

        The vector field is periodic in $θ$ with period $2π$. Numerical solutions
        retain an unwrapped angle, which can grow beyond a single revolution.

        The mechanical energy is

        $$ E = mgl(1 - \cos θ) + \frac{1}{2}ml²ω² $$

        The energy at the upper equilibrium is $E* = 2mgl$. To restrict the motion
        below this energy, choose initial conditions satisfying

        $$\begin{aligned}
            E₀ &< E* \\
            1 + \cos θ₀ &> \frac{l}{2g}ω₀² \\
            |ω₀| &< \sqrt{2(g/l)(1 + \cos θ₀)}
        \end{aligned}$$

        This restriction is optional and is not enforced by the generator. The
        equations also describe rotations that cross the upper position.

    References:
        - Neural Continuous-Discrete State Space Models <br>
          Abdul Fatir Ansari, Alvin Heng, Andre Lim, Harold Soh <br>
          Proceedings of the 40th International Conference on Machine Learning <br>
          <https://proceedings.mlr.press/v202/ansari23a.html> <br>
          <https://github.com/clear-nus/NCDSSM>
        - Deep Variational Bayes Filters: Unsupervised Learning of State Space Models from Raw Data <br>
          Maximilian Karl, Maximilian Soelch, Justin Bayer, Patrick van der Smagt <br>
          ICLR 2017 <br>
          <https://openreview.net/forum?id=HyTqHL5xg>
        - Deep Rao-Blackwellised Particle Filters for Time Series Forecasting <br>
          Richard Kurle, Syama Sundar Rangapuram, Emmanuel de Bézenac, Stephan Günnemann, Jan Gasthaus <br>
          NeurIPS 2020 <br>
          <https://proceedings.neurips.cc/paper/2020/hash/afb0b97df87090596ae7c503f60bb23f-Abstract.html>
          <https://dl.acm.org/doi/10.5555/3495724.3497013>
    """

    _: KW_ONLY

    g: float = 9.81
    r"""Gravitational acceleration."""
    length: float = 1.0
    r"""Length of the pendulum."""
    mass: float = 1.0
    r"""Mass of the pendulum."""
    gamma: float = 0.25
    r"""Damping coefficient."""
    theta0: float = np.pi
    r"""Initial angle."""
    omega0: float = 4.0
    r"""Initial angular velocity."""
    observation_noise_dist: RV = field(
        default_factory=lambda: univariate_normal(loc=0, scale=0.05)
    )
    r"""Noise distribution."""
    initial_state_dist: RV = field(
        default_factory=lambda: univariate_normal(loc=0, scale=1)
    )
    r"""Noise distribution."""

    def _get_initial_state_impl(self, *, size: Size = ()) -> NDArray:
        r"""Generate one or more initial states $y₀$."""
        p = self.initial_state_dist
        theta0 = self.theta0 + p.rvs(size=size, random_state=self.rng).clip(-2, +2)
        omega0 = self.omega0 * p.rvs(size=size, random_state=self.rng).clip(-2, +2)
        return np.stack([theta0, omega0], axis=-1)

    def _make_observations_impl(self, state: NDArray, /) -> NDArray:
        r"""Create observations from the solution."""
        # add observation noise
        p = self.observation_noise_dist
        return state + p.rvs(size=state.shape, random_state=self.rng)

    # Float[...], Float[..., 2] -> Float[..., 2]
    def system(self, t: ArrayLike, state: ArrayLike) -> NDArray:
        r"""Vector field of the pendulum.

        Supported input and output shapes include:

        - `[(...,), (2,)] -> (..., 2)`
        - `[(), (..., 2)] -> (..., 2)`
        """
        t = np.asarray(t)
        y = np.asarray(state)
        theta = y[..., 0]
        omega = y[..., 1]

        alpha = self.g / self.length
        beta = self.gamma / self.mass

        new_state = np.stack(
            [
                omega,
                -alpha * np.sin(theta) - beta * omega,
            ]
        )
        return np.einsum("..., ...d -> ...d", np.ones_like(t), new_state)


class DampedPendulumXY(DampedPendulum):
    r"""Damped pendulum simulation.

    This variant returns only Cartesian coordinates.
    """

    def _make_observations_impl(
        self, state: NDArray, /, *, noise: float = 0.05
    ) -> NDArray:
        r"""Create observations from the solution.

        Noise is automatically scaled by the length of the pendulum.
        """
        theta = state[..., 0]
        x = self.length * np.sin(theta)
        y = -self.length * np.cos(theta)
        loc = np.stack([x, y], axis=-1)

        # Random noise on angle
        # sample from truncated normal distribution
        # cf. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
        loc_min, loc_max = -self.length, +self.length
        noise *= self.length
        lower = (loc_min - loc) / noise
        upper = (loc_max - loc) / noise
        result = truncnorm.rvs(
            lower, upper, loc=loc, scale=noise, random_state=self.rng
        )
        return np.asarray(result)

    def validate_observations(self, values: NDArray, /) -> None:
        r"""Validate constraints on the observations."""
        if values.min() < -self.length:
            raise ValueError(f"Minimum value {values.min()} < -{self.length}")
        if values.max() > +self.length:
            raise ValueError(f"Maximum value {values.max()} > +{self.length}")
