r"""Dampened Pendulum Simulation.

References:
    - https://en.wikipedia.org/wiki/Pendulum_(mechanics)
"""

__all__ = ["DampedPendulum", "DampedPendulumXY"]

from dataclasses import KW_ONLY, dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm as univariate_normal, truncnorm

from tsdm.random.distributions import RV
from tsdm.random.generators.base import IVP_GeneratorBase
from tsdm.types.aliases import Size


@dataclass
class DampedPendulum(IVP_GeneratorBase):
    r"""Dampened Pendulum Simulation.

    The dampended pendulum is an autonomous system with two degrees of freedom.

    .. math::
        dθ/dt = ω
        dω/dt = -(g/l)⋅sin(θ) - (γ/m)⋅ω

    Note: (Second equilibrium)
        Usually the top position is an unstable equilibrium.
        For the dampened pendulum, the jacobian of the vector field at θ=π, ω=0 is

        .. math::
            J = [[0, 1], [-(g/l)\cos θ, -γ/m]]
              = [[0, 1], [g/l, -γ/m]]

        Applying the formula for the eigenvalues of a 2x2 matrix,

        .. math:: λ = ½(tr J ± √{(tr J)² - 4⋅det J})

        We get

        .. math:: λ = ½(-γ/m ± √{γ²/m² + 4g/l})

        The equilibrium is stable if Re(λ)<0, so in this case iff

        .. math::
            ½(-γ/m ± √{γ²/m² + 4g/l}) < 0
            ⟺ √{γ²/m² + 4g/l}) < γ/m
            ⟺ 4g/l < 0

        which is obviously false. Hence, the top position is a stable equilibrium.

    Note: (periodicity)
        These equations are periodic with a period of 2π.
        in particular, quite annoyingly, when solving numerically we can
        get rather large values for θ. This could be avoided by using a
        different coordinate system, but we don't do that here.

    Warning:
        This variant leads to un-physical results.
        This is because the equations only work if the pendulum cannot cross
        the top position.

    Note: (critical energy)
        The energy of the pendulum at position θ and velocity ω is

        .. math:: E = m⋅g⋅l⋅(1 - cos θ) + ½⋅m⋅l²⋅ω²

        The critical energy is the energy at the top position, i.e. θ=π, ω=0.

        Therefore, we need to choose the initial conditions such that the
        energy is below the critical energy.

        .. math:: E* > E₀
            ⟺ 2⋅m⋅g⋅l > m⋅g⋅l⋅(1 - cos θ₀) + ½⋅m⋅l²⋅ω₀²
            ⟺ 1 + cos θ₀ > ½⋅(l/g)⋅ω₀²
            ⟺ ω₀ < √{ 2(g/l)(1 + cos θ₀) }

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
        r"""Generate (multiple) initial state(s) y₀."""
        p = self.initial_state_dist
        theta0 = self.theta0 + p.rvs(size=size, random_state=self.rng).clip(-2, +2)
        omega0 = self.omega0 * p.rvs(size=size, random_state=self.rng).clip(-2, +2)
        return np.stack([theta0, omega0], axis=-1)

    def _make_observations_impl(self, y: NDArray, /) -> NDArray:
        r"""Create observations from the solution."""
        # add observation noise
        p = self.observation_noise_dist
        return y + p.rvs(size=y.shape, random_state=self.rng)

    def system(self, t: ArrayLike, state: ArrayLike) -> NDArray:
        r"""Vector field of the pendulum.

        .. signature:: ``[(...,), (..., 2) -> (..., 2)``

        sub-signatures:
            - ``[(...,), (2, ) -> (..., 2)``
            - ``[(,), (..., 2) -> (..., 2)``
        """
        t = np.asarray(t)
        y = np.asarray(state)
        theta = y[..., 0]
        omega = y[..., 1]

        alpha = self.g / self.length
        beta = self.gamma / self.mass

        new_state = np.stack([
            omega,
            -alpha * np.sin(theta) - beta * omega,
        ])
        return np.einsum("..., ...d -> ...d", np.ones_like(t), new_state)


class DampedPendulumXY(DampedPendulum):
    r"""Dampened Pendulum Simulation.

    This variant returns only cartesian coordinates.
    """

    def _make_observations_impl(self, y: NDArray, /, *, noise: float = 0.05) -> NDArray:
        r"""Create observations from the solution.

        Noise is automatically scaled by the length of the pendulum.
        """
        theta = y[..., 0]
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
        r"""Validate constraints on the parameters."""
        if values.min() < -self.length:
            raise ValueError(f"Minimum value {values.min()} < -{self.length}")
        if values.max() > +self.length:
            raise ValueError(f"Maximum value {values.max()} > +{self.length}")
