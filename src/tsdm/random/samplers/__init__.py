r"""Random Samplers.

Note:
    Samplers are used to randomly select **indices** that can be used to select data.
    For methods that randomly select data from the data source directly, see `tsdm.random.generators`.
"""

__all__ = [
    # Constants
    "SAMPLERS",
    # ABC & Protocols
    "BaseSampler",
    "Sampler",
    # Classes
    "HierarchicalSampler",
    "IntervalSampler",
    "RandomSampler",
    "SequenceSampler",
    "SlidingSampler",
    # Functions
    "compute_grid",
]

from tsdm.random.samplers._samplers_deprecated import IntervalSampler, SequenceSampler
from tsdm.random.samplers.base import (
    BaseSampler,
    HierarchicalSampler,
    RandomSampler,
    Sampler,
    SlidingSampler,
    compute_grid,
)

SAMPLERS: dict[str, type[Sampler]] = {
    "HierarchicalSampler"  : HierarchicalSampler,
    "IntervalSampler"      : IntervalSampler,
    "RandomSampler"        : RandomSampler,
    "SequenceSampler"      : SequenceSampler,
    "SlidingSampler"       : SlidingSampler,
}  # fmt: skip
r"""Dictionary of all available samplers."""
