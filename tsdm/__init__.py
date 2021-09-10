#!usr/bin/env python
r"""Time Series Datasets and Models (TSDM).

Provides
  1. Facility to import some commonly used time series datasets
  2. Facility to import some commonly used time series models
  3. Facility to preprocess time series datasets

More complicated examples:

Random Search / Grid Search Hyperparameter optimization with nested cross-validation
split on a slurm cluster.

General idea:

1. Datasets should store data in "original" / "clean" / "pure form"
    - all kinds of data types allowed
    - all data types must support NaN values (-> pandas Int64 and StringDType !)
2. DataLoaders perform 2 tasks
    1. Encoding the data into pure float tensors
        - Consider different kinds of encoding
    2. Creating generator objects
        - random sampling from dataset
        - batching of random samples
        - caching?

"""

import logging
from pathlib import Path
from typing import Final

from . import (
    config,
    datasets,
    encoders,
    generators,
    losses,
    models,
    optimizers,
    plot,
    random,
    trainers,
    util,
)

with open(Path(__file__).parent.joinpath("VERSION"), "r", encoding="utf8") as file:
    __version__ = file.read()
    r"""The version number of the tsdm package"""

logger = logging.getLogger(__name__)
__all__: Final[list[str]] = ["__version__"] + [
    "config",
    "datasets",
    "generators",
    "encoders",
    "losses",
    "models",
    "optimizers",
    "plot",
    "random",
    "trainers",
    "util",
]
