r"""Synthetic Data Generators.

Note:
    Generators are used to create synthetic data.
    For methods to randomly select from pre-existing data, see `tsdm.random.samplers`.
"""

__all__ = [
    # CONSTANTS
    "GENERATORS",
    "IVP_SOLVERS",
    # Classes
    "BouncingBall",
    "DampedPendulum",
    "DampedPendulumXY",
    "LotkaVolterra",
    "SIR",
]
from . import base
from .base import *  # ruff: ignore[F403]
from .bouncing_ball import BouncingBall
from .dampened_pendulum import DampedPendulum, DampedPendulumXY
from .lotka_volterra import LotkaVolterra
from .sir_model import SIR

__all__ += base.__all__

GENERATORS: dict[str, type[base.IVP_Generator]] = {
    "BouncingBall"     : BouncingBall,
    "DampedPendulum"   : DampedPendulum,
    "DampedPendulumXY" : DampedPendulumXY,
    "LotkaVolterra"    : LotkaVolterra,
    "SIR"              : SIR,
}  # fmt: skip
r"""Dictionary of all available generators."""

IVP_SOLVERS: dict[str, base.IVP_Solver] = {
    "solve_ivp" : base.solve_ivp,
}  # fmt: skip
r"""Dictionary of all available IVP solvers."""
