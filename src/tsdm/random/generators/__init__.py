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
    "Distribution",
    "TimeSeriesDistribution",
    "IVP_Generator",
    "IVP_Solver",
    # Generators
    "BouncingBall",
    "DampedPendulum",
    "LoktaVolterra",
]

from tsdm.random.generators._generators import (
    Distribution,
    Generator,
    IVP_Generator,
    IVP_Solver,
    TimeSeriesDistribution,
    TimeSeriesGenerator,
)
from tsdm.random.generators.bouncing_ball import BouncingBall
from tsdm.random.generators.dampened_pendulum import DampedPendulum
from tsdm.random.generators.lotka_volterra import LoktaVolterra

GENERATORS: dict[str, type[TimeSeriesGenerator]] = {
    "DampedPendulum": DampedPendulum,
    "LoktaVolterra": LoktaVolterra,
    "BouncingBall": BouncingBall,
}
r"""Dictionary of all available generators."""
