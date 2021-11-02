r"""Generators for synthetic time series datasets.

Contrary to tsdm.datasets, which contains static, real-world datasets, this module
contains generators for synthetic datasets. By design each generators consists of

- Configuration parameters, e.g. number of dimensions etc.
- Allows sampling from the data ground truth distribution p(x,y)
- Allows estimating the Bayes Error, i.e. the best performance possible on the dataset.
"""
# TODO: add some generators

from __future__ import annotations

__all__ = [
    # Constants
    "Generator",
    "GENERATORS",
]

import logging
from typing import Any, Final

__logger__ = logging.getLogger(__name__)

Generator = Any
r"""Type hint for generators."""

GENERATORS: Final[dict[str, Generator]] = {}
r"""Dictionary of all available generators."""
