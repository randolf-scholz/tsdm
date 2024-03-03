"""Generator for helix motion."""

__all__ = ["Helix"]

from dataclasses import dataclass, field

import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
from scipy.stats import multivariate_normal
from typing_extensions import Any

from tsdm.random.generators._generators import IVP_GeneratorBase
from tsdm.random.stats.distributions import Distribution
from tsdm.types.aliases import SizeLike


@dataclass
class Helix(IVP_GeneratorBase[NDArray]):
    r"""Helix Motion Simulation.

    The particle moves at constant speed in the given direction, and follows a circular path
    in the orthogonal plane.

    In the standard setting, the motion is given by:

    .. math::
         x(t) = r⋅cos(ωt + φ)
         y(t) = r⋅sin(ωt + φ)
         z(t) = ρ⋅t

     Corresponding to the following system of differential equations:

    .. math::
        \dot{x} = -ω⋅y
        \dot{y} = +ω⋅x
        \dot{z} = ρ
    """

    angular_velocity: float = 1.0
    """Angular velocity."""
    phase: float = 0.0
    """Initial angle in radians."""
    radius: float = 1.0
    """Radius of the helix."""
    pitch: float = 1.0
    """Pitch of the helix."""
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Direction of the helix."""
    observation_noise: Distribution = field(
        default_factory=lambda: multivariate_normal(mean=np.zeros(3), cov=0.1)
    )
    """Noise distribution."""
    seed: Generator = field(default_factory=default_rng)
    """Random number generator."""

    def __post_init__(self) -> None:
        """Post-initialization hook."""
        # extend to orthogonal basis
        z = np.array(self.direction)
        x = np.array([-z[1], z[0], 0.0])
        y = np.cross(z, x)
        self.x = x / np.linalg.norm(x)
        self.y = y / np.linalg.norm(y)
        self.z = z / np.linalg.norm(z)

        # get transformation matrix w.r.t. standard basis
        self.Q = np.stack([self.x, self.y, self.z], axis=-1)
        self.Qt = self.Q.T  # cache transpose

    def _get_initial_state_impl(self, *, size: SizeLike = ()) -> NDArray:
        pass

    def _make_observations_impl(self, sol: NDArray, /) -> NDArray:
        """Additive noise."""
        return sol + self.observation_noise.rvs(size=sol.shape)

    def system(self, t: Any, state: NDArray) -> NDArray:
        """System function."""
        # 1. transform to basis
        state = np.einsum("...i,ij->...j", state, self.T)
        # 2. extract variables
        x = state[..., 0]
        y = state[..., 1]
        z = state[..., 2]
        # get the current angle (-> new phase)
        theta = np.arctan2(y, x)

        # 3. compute future state
        x = self.radius * np.cos(self.angular_velocity * t + theta)
        y = self.radius * np.sin(self.angular_velocity * t + theta)
        z = self.pitch * t + z

        result = np.stack([x, y, z], axis=-1)
        # 4. Transform back to the standard basis
        return np.einsum("...i,ij->...j", result, self.Qt)