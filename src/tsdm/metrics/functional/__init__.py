r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # Types
    "FunctionalMetric",
    # Constants
    "FunctionalMetrics",
]

import logging
from collections.abc import Callable
from typing import Final

from torch import Tensor

from tsdm.util.types import LookupTable

__logger__ = logging.getLogger(__name__)


FunctionalMetric = Callable[..., Tensor]
r"""Type hint for functional metrics."""

FunctionalMetrics: Final[LookupTable[FunctionalMetric]] = {}
r"""Dictionary of all available functional metrics."""
