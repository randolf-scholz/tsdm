"""Generator for helix motion."""

__all__ = ["Helix"]

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal
from typing_extensions import Any

from tsdm.random.distributions import RV
from tsdm.random.generators.base import IVP_GeneratorBase
from tsdm.types.aliases import Size


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
    r"""Angular velocity."""
    phase: float = 0.0
    r"""Initial angle in radians."""
    radius: float = 1.0
    r"""Radius of the helix."""
    pitch: float = 1.0
    r"""Pitch of the helix."""
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    r"""Direction of the helix."""

    def __post_init__(self) -> None:
        r"""Post-initialization hook."""
        # construct ONB using revised Frisvad algorithm
        # REF: Building an Orthonormal Basis, Revisited https://jcgt.org/published/0006/01/01/
        d = np.array(self.direction)
        x, y, z = d

        s = np.copysign(1.0, z)
        a = -1.0 / (s + z)
        b = x * y * a
        u = np.array([1.0 + s * a * x**2, s * b, -s * x])
        v = np.array([b, s + a * y**2, -a * y])

        # get transformation matrix w.r.t. standard basis
        self.Q = np.stack([u, v, d], axis=-1)
        self.Q /= np.linalg.norm(self.Q, axis=0)
        self.Qt = self.Q.T  # cache transpose

    @property
    def initial_state_dist(self) -> RV:
        r"""Noise distribution."""
        return multivariate_normal(mean=np.zeros(3), cov=0.1)

    @property
    def observation_noise_dist(self) -> RV:
        r"""Noise distribution."""
        return multivariate_normal(mean=np.zeros(3), cov=0.1)

    def _get_initial_state_impl(self, *, size: Size = ()) -> NDArray:
        p = self.initial_state_dist
        return p.rvs(size=size, random_state=self.rng)

    def _make_observations_impl(self, sol: NDArray, /) -> NDArray:
        r"""Additive noise."""
        p = self.observation_noise_dist
        return sol + p.rvs(size=sol.shape, random_state=self.rng)

    def system(self, t: Any, state: NDArray) -> NDArray:
        r"""System function."""
        # 1. transform to basis
        state = np.einsum("...i,ij->...j", state, self.Q)
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
