r"""Samplers for from sampling datasets and timeseries.

Note:
    These are often used to sample keys used for indexing into a dataset,
    rather than sampling directly from the dataset itself.
"""

__all__ = [
    # submodules
    "base",
    "functional",
    # Constants
    "SAMPLERS",
    # Classes
    "HierarchicalSampler",
    "HierarchicalDataset",
    "RandomSampler",
    "SlidingWindowSampler",
    # Functions
    "compute_grid",
]

from . import base, functional
from .base import *  # ruff: ignore[F403]
from .functional import *  # ruff: ignore[F403]
from .hierarchical_sampler import HierarchicalDataset, HierarchicalSampler
from .random_sampler import RandomSampler
from .sliding_window_sampler import SlidingWindowSampler, compute_grid

__all__ += base.__all__
__all__ += functional.__all__

SAMPLERS: dict[str, type[base.Sampler]] = {
    "HierarchicalSampler"  : HierarchicalSampler,
    "RandomSampler"        : RandomSampler,
    "SlidingWindowSampler" : SlidingWindowSampler,
}  # fmt: skip
r"""Mapping from public sampler names to their implementation classes."""
