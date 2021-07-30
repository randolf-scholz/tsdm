#!usr/bin/env python
r"""Time Series Datasets and Models (TSDM).

Provides
  1. Facility to import some commonly used time series datasets
  2. Facility to import some commonly used time series models
  3. Facility to preprocess time series datasets
"""

from pathlib import Path

from tsdm import config, datasets, generators, losses, models, plot, util

with open(Path(__file__).parent.joinpath("VERSION"), "r") as file:
    __version__ = file.read()
    r"""The version number of the tsdm package"""

del Path

__all__ = [
    "__version__",
    "config",
    "datasets",
    "generators",
    "losses",
    "models",
    "util",
    "plot",
]
