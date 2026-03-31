r"""Statistical functions for random variables."""

__all__ = [
    # Sub-Packages
    "generators",
    "samplers",
    "stats",
    # Functions
    "random_data",
    "sample_timestamps",
    "sample_timedeltas",
]

from . import generators, samplers, stats
from ._random import random_data, sample_timedeltas, sample_timestamps
