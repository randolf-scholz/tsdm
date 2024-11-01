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
    "RandomSampler",
    "SequenceSampler",
    "SlidingSampler",
    # Functions
    "compute_grid",
]

from tsdm.random.samplers._samplers_deprecated import SequenceSampler
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
    "RandomSampler"        : RandomSampler,
    "SequenceSampler"      : SequenceSampler,
    "SlidingSampler"       : SlidingSampler,
}  # fmt: skip
r"""Dictionary of all available samplers."""
