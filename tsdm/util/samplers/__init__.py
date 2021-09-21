r"""Module Summary Line.

Module description
"""  # pylint: disable=line-too-long # noqa

from __future__ import annotations

import logging
from typing import Final, Type

from torch.utils import data as torch_utils_data

from tsdm.util.samplers.samplers import SequenceSampler, SliceSampler, TimeSliceSampler

LOGGER = logging.getLogger(__name__)

__all__: Final[list[str]] = [
    "SliceSampler",
    "TimeSliceSampler",
    "SequenceSampler",
]

Sampler = Type[torch_utils_data.Sampler]
r"""Type hint for samplers."""

SAMPLERS: Final[dict[str, Sampler]] = {
    "SliceSampler": SliceSampler,
    "TimeSliceSampler": TimeSliceSampler,
    "SequenceSampler": SequenceSampler,
}
r"""Dictionary containing all available samplers."""
