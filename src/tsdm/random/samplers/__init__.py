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
    "SlidingWindowSampler",
    # Functions
    "compute_grid",
]

from tsdm.random.samplers.base import BaseSampler, RandomSampler, Sampler
from tsdm.random.samplers.hierarchical_sampler import HierarchicalSampler
from tsdm.random.samplers.sliding_window_sampler import (
    SlidingWindowSampler,
    compute_grid,
)

SAMPLERS: dict[str, type[Sampler]] = {
    "HierarchicalSampler"  : HierarchicalSampler,
    "RandomSampler"        : RandomSampler,
    "SlidingSampler"       : SlidingWindowSampler,
}  # fmt: skip
r"""Dictionary of all available samplers."""
