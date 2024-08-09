r"""Duffing Oszillator Simulation."""

__all__ = ["DuffingOszillator"]

from dataclasses import KW_ONLY, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tsdm.random.generators.base import IVP_GeneratorBase


@dataclass
class DuffingOszillator(IVP_GeneratorBase):
    r"""Duffing Oszillator Simulation.

    References:
        - https://en.wikipedia.org/wiki/Duffing_equation
    """

    _: KW_ONLY

    alpha: float = 1.0
    r"""Linear stiffness."""
    beta: float = 1.0
    r"""Nonlinear stiffness."""
    delta: float = 0.0
    r"""Damping coefficient  (default: undamped)."""
    gamma: float = 0.0
    r"""Amplitude of periodic driving force (default: no driving force)."""
    omega: float = 1.0
    r"""Angular frequency of periodic driving force."""

    def system(self, t: ArrayLike, state: ArrayLike) -> NDArray:
        T = np.asarray(t)
        S = np.asarray(state)

        x = S[..., 0]
        p = S[..., 1]
        return np.stack(
            [
                p,
                -self.alpha * x
                - self.beta * x**3
                + self.gamma * np.cos(self.omega * T),
            ],
            axis=-1,
        )
