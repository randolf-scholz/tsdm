r"""Random Samplers."""

__all__ = [
    # Constants
    "Sampler",
    "SAMPLERS",
    # ABC
    "BaseSampler",
    "BaseSamplerMetaClass",
    # Classes
    "CollectionSampler",
    "HierarchicalSampler",
    "IntervalSampler",
    "SequenceSampler",
    "SliceSampler",
    "SlidingWindowSampler",
    # Functions
    "compute_grid",
]

from typing import Final, TypeAlias

from torch.utils.data import Sampler as TorchSampler

from tsdm.random.samplers._samplers import (
    BaseSampler,
    BaseSamplerMetaClass,
    CollectionSampler,
    HierarchicalSampler,
    IntervalSampler,
    SequenceSampler,
    SliceSampler,
    SlidingWindowSampler,
    compute_grid,
)

Sampler: TypeAlias = TorchSampler
r"""Type hint for samplers."""

SAMPLERS: Final[dict[str, type[Sampler]]] = {
    "SliceSampler": SliceSampler,
    # "TimeSliceSampler": TimeSliceSampler,
    "SequenceSampler": SequenceSampler,
    "CollectionSampler": CollectionSampler,
}
r"""Dictionary of all available samplers."""

del Final, TypeAlias, TorchSampler
