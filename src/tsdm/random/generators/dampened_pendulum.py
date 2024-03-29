"""Dampened Pendulum Simulation.

References:
    - https://en.wikipedia.org/wiki/Pendulum_(mechanics)
"""

__all__ = ["DampedPendulum", "DampedPendulumXY"]

from dataclasses import KW_ONLY, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm as univariate_normal, truncnorm

from tsdm.random.generators._generators import IVP_GeneratorBase
from tsdm.random.stats.distributions import Distribution
from tsdm.types.aliases import SizeLike


@dataclass
class DampedPendulum(IVP_GeneratorBase[NDArray]):
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
    observation_noise: Distribution = univariate_normal(loc=0, scale=0.05)
    """Noise distribution."""
    parameter_noise: Distribution = univariate_normal(loc=0, scale=1)
    """Noise distribution."""

    def _get_initial_state_impl(self, size: SizeLike = ()) -> NDArray:
        """Generate (multiple) initial state(s) y₀."""
        theta0 = self.theta0 + self.parameter_noise.rvs(size=size).clip(-2, +2)
        omega0 = self.omega0 * self.parameter_noise.rvs(size=size).clip(-2, +2)
        return np.stack([theta0, omega0], axis=-1)

    def _make_observations_impl(self, y: NDArray, /) -> NDArray:
        """Create observations from the solution."""
        # add observation noise
        return y + self.observation_noise.rvs(size=y.shape)

    def system(self, t: ArrayLike, x: ArrayLike) -> NDArray:
        """Vector field of the pendulum.

        .. Signature:: ``[(...,), (..., 2) -> (..., 2)``

        sub-signatures:
            - ``[(...,), (2, ) -> (..., 2)``
            - ``[(,), (..., 2) -> (..., 2)``
        """
        t = np.asarray(t)
        state = np.asarray(x)
        theta = state[..., 0]
        omega = state[..., 1]

        alpha = self.g / self.length
        beta = self.gamma / self.mass

        new_state = np.stack([
            omega,
            -alpha * np.sin(theta) - beta * omega,
        ])
        return np.einsum("..., ...d -> ...d", np.ones_like(t), new_state)


class DampedPendulumXY(DampedPendulum):
    """Dampened Pendulum Simulation.

    This variant returns only cartesian coordinates.
    """

    def _make_observations_impl(self, y: NDArray, /, *, noise: float = 0.05) -> NDArray:
        """Create observations from the solution.

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
        noise = noise * self.length
        lower = (loc_min - loc) / noise
        upper = (loc_max - loc) / noise
        return truncnorm.rvs(lower, upper, loc=loc, scale=noise)

    def validate_observations(self, values: NDArray, /) -> None:
        """Validate constraints on the parameters."""
        assert values.min() >= -self.length and values.max() <= +self.length
