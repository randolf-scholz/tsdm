r"""Statistical functions for random variables."""


__all__ = [
    # Sub-Packages
    "generators",
    "samplers",
    "stats",
    # Functions
    "sample_timestamps",
    "sample_timedeltas",
]


import logging

from tsdm.random import generators, samplers, stats
from tsdm.random._random import sample_timedeltas, sample_timestamps

__logger__ = logging.getLogger(__name__)
