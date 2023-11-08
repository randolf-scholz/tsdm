r"""Random Samplers.

Note:
    Samplers are used to randomly select **indices** that can be used to select data.
    For methods that randomly select data from the data source directly, see `tsdm.random.generators`.
"""

__all__ = [
    # Constants
    "SAMPLERS",
    # ABC
    "Sampler",
    "BaseSampler",
    # Classes
    "HierarchicalSampler",
    "RandomSampler",
    "SlidingWindowSampler",
    "CollectionSampler",
    "SliceSampler",
    # Functions
    "compute_grid",
]

from typing import Final, TypeAlias

from torch.utils.data import Sampler as TorchSampler

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
