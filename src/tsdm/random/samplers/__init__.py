r"""Random Samplers.

Note:
    Samplers are used to randomly select samples from pre-existing data.
    For methods to randomly generate data, see `tsdm.random.generators`.
"""

__all__ = [
    # Constants
    "SAMPLERS",
    # ABC
    "Sampler",
    "BaseSampler",
    # Classes
    "CollectionSampler",
    "HierarchicalSampler",
    "IntervalSampler",
    "RandomSampler",
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
    CollectionSampler,
    HierarchicalSampler,
    IntervalSampler,
    RandomSampler,
    Sampler,
    SequenceSampler,
    SliceSampler,
    SlidingWindowSampler,
    compute_grid,
)

SAMPLERS: Final[dict[str, type[Sampler]]] = {
    "CollectionSampler": CollectionSampler,
    "HierarchicalSampler": HierarchicalSampler,
    "IntervalSampler": IntervalSampler,
    "SequenceSampler": SequenceSampler,
    "SliceSampler": SliceSampler,
    "SlidingWindowSampler": SlidingWindowSampler,
    "RandomSampler": RandomSampler,
}
r"""Dictionary of all available samplers."""

del Final, TypeAlias, TorchSampler
