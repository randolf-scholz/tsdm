"""Duffing Oszillator Simulation."""

__all__ = ["DuffingOszillator"]

from dataclasses import KW_ONLY, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tsdm.random.generators._generators import IVP_GeneratorBase


@dataclass
class DuffingOszillator(IVP_GeneratorBase[NDArray]):
    """Duffing Oszillator Simulation.

    References:
        - https://en.wikipedia.org/wiki/Duffing_equation
    """

    _: KW_ONLY

    alpha: float = 1.0
    """Linear stiffness."""
    beta: float = 1.0
    """Nonlinear stiffness."""
    delta: float = 0.0
    """Damping coefficient  (default: undamped)."""
    gamma: float = 0.0
    """Amplitude of periodic driving force (default: no driving force)."""
    omega: float = 1.0
    """Angular frequency of periodic driving force."""

    def system(self, t: Any, state: ArrayLike) -> NDArray:
        x = state[..., 0]
        p = state[..., 1]
        return np.stack(
            [
                p,
                -self.alpha * x
                - self.beta * x**3
                + self.gamma * np.cos(self.omega * t),
            ],
            axis=-1,
        )
