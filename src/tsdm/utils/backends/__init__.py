"""Utilities for backends.

Currently, supports pandas / numpy / torch.
"""

__all__ = [
    # Constants
    "KERNELS",
    # Type aliases
    "Backend",
    # Functions
    "get_backend",
    "where_kernel",
    "clip_kernel",
]

from tsdm.utils.backends.numerical import (
    Backend,
    clip_kernel,
    get_backend,
    where_kernel,
)

KERNELS = {
    "where": where_kernel,
    "clip": clip_kernel,
}
