r"""Random Samplers.

Note:
    Samplers are used to randomly select **indices** that can be used to select data.
    For methods that randomly select data from the data source directly, see `tsdm.random.generators`.
"""

__all__ = [
    # submodules
    "functional",
    # Constants
    "SAMPLERS",
    # ABC & Protocols
    "BaseSampler",
    "Sampler",
    # Classes
    "MappingDataset",
    "HierarchicalSampler",
    "RandomSampler",
    "SlidingWindowSampler",
    # Functions
    "compute_grid",
]


from . import functional
from .base import BaseSampler, RandomSampler, Sampler
from .hierarchical_sampler import HierarchicalSampler, MappingDataset
from .sliding_window_sampler import SlidingWindowSampler, compute_grid

SAMPLERS: dict[str, type[Sampler]] = {
    "HierarchicalSampler"  : HierarchicalSampler,
    "RandomSampler"        : RandomSampler,
    "SlidingWindowSampler" : SlidingWindowSampler,
}  # fmt: skip
r"""Dictionary of all available samplers."""
