r"""TODO: Module Summary Line.

TODO: Module description
"""

from __future__ import annotations

__all__ = [
    # Constants
    "Sampler",
    "SAMPLERS",
    # Classes
    "SliceSampler",
    "TimeSliceSampler",
    "SequenceSampler",
]

import logging
from typing import Final

from torch.utils import data as torch_utils_data

from tsdm.util.samplers.samplers import SequenceSampler, SliceSampler, TimeSliceSampler

LOGGER = logging.getLogger(__name__)

Sampler = torch_utils_data.Sampler
r"""Type hint for samplers."""

SAMPLERS: Final[dict[str, type[Sampler]]] = {
    "SliceSampler": SliceSampler,
    "TimeSliceSampler": TimeSliceSampler,
    "SequenceSampler": SequenceSampler,
}
r"""Dictionary of all available samplers."""
