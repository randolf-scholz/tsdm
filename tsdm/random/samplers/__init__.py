r"""Random Samplers."""

__all__ = [
    # Constants
    "Sampler",
    "SAMPLERS",
    # Classes
    "SliceSampler",
    # "TimeSliceSampler",
    "SequenceSampler",
    "CollectionSampler",
]

import logging
from typing import Final

from torch.utils import data as torch_utils_data

from tsdm.random.samplers.samplers import (
    CollectionSampler,
    SequenceSampler,
    SliceSampler,
)

__logger__ = logging.getLogger(__name__)

Sampler = torch_utils_data.Sampler
r"""Type hint for samplers."""

SAMPLERS: Final[dict[str, type[Sampler]]] = {
    "SliceSampler": SliceSampler,
    # "TimeSliceSampler": TimeSliceSampler,
    "SequenceSampler": SequenceSampler,
    "CollectionSampler": CollectionSampler,
}
r"""Dictionary of all available samplers."""
