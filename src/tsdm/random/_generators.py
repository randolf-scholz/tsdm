r"""Generators for synthetic time series dataset.

Contrary to `tsdm.dataset`, which contains static, real-world dataset, this module
contains generators for synthetic dataset. By design each generator consists of

- Configuration parameters, e.g. number of dimensions etc.
- Allows sampling from the data ground truth distribution p(x,y)
- Allows estimating the Bayes Error, i.e. the best performance possible on the dataset.
"""

__all__ = [
    # Constants
    "Generator",
    "GENERATORS",
]

from typing import Protocol, runtime_checkable


@runtime_checkable
class Generator(Protocol):
    r"""Protocol for generators."""


GENERATORS: dict[str, Generator] = {}
r"""Dictionary of all available generators."""
