r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # Types
    "ModularMetric",
    # Constants
    "ModularMetrics",
]

import logging
from typing import Final

from torch import nn

__logger__ = logging.getLogger(__name__)


ModularMetric = nn.Module
r"""Type hint for modular losses."""

ModularMetrics: Final[dict[str, type[nn.Module]]] = {}
r"""Dictionary of all available modular metrics."""
