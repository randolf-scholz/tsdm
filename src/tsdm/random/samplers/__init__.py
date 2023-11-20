r"""Random Samplers.

Note:
    Samplers are used to randomly select **indices** that can be used to select data.
    For methods that randomly select data from the data source directly, see `tsdm.random.generators`.
"""

__all__ = [
    # Constants
    "SAMPLERS",
    # Protocols
    "Sampler",
    # ABC
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

from tsdm.random.samplers._samplers import (
    BaseSampler,
    HierarchicalSampler,
    RandomSampler,
    Sampler,
    SlidingWindowSampler,
    compute_grid,
)
from tsdm.random.samplers._samplers_deprecated import (
    CollectionSampler,
    IntervalSampler,
    SequenceSampler,
    SliceSampler,
)

SAMPLERS: dict[str, type[Sampler]] = {
    "CollectionSampler": CollectionSampler,
    "HierarchicalSampler": HierarchicalSampler,
    "IntervalSampler": IntervalSampler,
    "SequenceSampler": SequenceSampler,
    "SliceSampler": SliceSampler,
    "SlidingWindowSampler": SlidingWindowSampler,
    "RandomSampler": RandomSampler,
}
r"""Dictionary of all available samplers."""
