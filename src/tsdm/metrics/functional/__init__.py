r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # Types
    "FunctionalMetric",
    # Constants
    "FunctionalMetrics",
]

from collections.abc import Callable
from typing import Final

from torch import Tensor

FunctionalMetric = Callable[..., Tensor]
r"""Type hint for functional metrics."""

FunctionalMetrics: Final[dict[str, FunctionalMetric]] = {}
r"""Dictionary of all available functional metrics."""
