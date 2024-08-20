r"""Synthetic Data Generators.

Note:
    Generators are used to create synthetic data.
    For methods to randomly select from pre-existing data, see `tsdm.random.samplers`.
"""

__all__ = [
    # CONSTANTS
    "GENERATORS",
    "IVP_SOLVERS",
    # ABCs & Protocols
    "IVP_Generator",
    "IVP_Solver",
    "ODE",
    "FrozenIVPSolver",
    "ScipyIVPSolver",
    "IVP_GeneratorBase",
    # Classes
    "BouncingBall",
    "DampedPendulum",
    "DampedPendulumXY",
    "LotkaVolterra",
    "SIR",
    # Functions
    "solve_ivp",
]

from tsdm.random.generators.base import (
    ODE,
    FrozenIVPSolver,
    IVP_Generator,
    IVP_GeneratorBase,
    IVP_Solver,
    ScipyIVPSolver,
    solve_ivp,
)
from tsdm.random.generators.bouncing_ball import BouncingBall
from tsdm.random.generators.dampened_pendulum import DampedPendulum, DampedPendulumXY
from tsdm.random.generators.lotka_volterra import LotkaVolterra
from tsdm.random.generators.sir_model import SIR

GENERATORS: dict[str, type[IVP_Generator]] = {
    "BouncingBall"     : BouncingBall,
    "DampedPendulum"   : DampedPendulum,
    "DampedPendulumXY" : DampedPendulumXY,
    "LotkaVolterra"    : LotkaVolterra,
    "SIR"              : SIR,
}  # fmt: skip
r"""Dictionary of all available generators."""

IVP_SOLVERS: dict[str, IVP_Solver] = {
    "solve_ivp" : solve_ivp,
}  # fmt: skip
r"""Dictionary of all available IVP solvers."""
