"""Synthetic Data Generators.

Note:
    Generators are used to create synthetic data.
    For methods to randomly select from pre-existing data, see `tsdm.random.samplers`.
"""


__all__ = [
    # CONSTANTS
    "GENERATORS",
    # Protocols
    "Generator",
    "TimeSeriesGenerator",
    "IVP_Generator",
    "IVP_Solver",
    # Generators
    "BouncingBall",
    "DampedPendulum",
    "LotkaVolterra",
    "SIR",
    # functions
    "solve_ivp",
]

from tsdm.random.generators._generators import (
    Generator,
    IVP_Generator,
    IVP_Solver,
    TimeSeriesGenerator,
    solve_ivp,
)
from tsdm.random.generators.bouncing_ball import BouncingBall
from tsdm.random.generators.dampened_pendulum import DampedPendulum
from tsdm.random.generators.lotka_volterra import LotkaVolterra
from tsdm.random.generators.sir_model import SIR

GENERATORS: dict[str, type[TimeSeriesGenerator]] = {
    "DampedPendulum": DampedPendulum,
    "LoktaVolterra": LotkaVolterra,
    "BouncingBall": BouncingBall,
    "SIR": SIR,
}
r"""Dictionary of all available generators."""
