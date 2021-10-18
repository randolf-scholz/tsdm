r"""#TODO add module summary line.

#TODO add module description.
"""

from __future__ import annotations

__all__ = [
    # Types
    "ModularMetric",
    # Constants
    "ModularMetrics",
]

import logging
from typing import Final

from torch import nn

from tsdm.util.types import LookupTable

LOGGER = logging.getLogger(__name__)


ModularMetric = nn.Module
r"""Type hint for modular losses."""

ModularMetrics: Final[LookupTable[type[ModularMetric]]] = {}
r"""Dictionary of all available modular metrics."""
