"""Utilities for backends.

Currently, supports pandas / numpy / torch.
"""

__all__ = [
    # Constants
    # "KERNELS",
    # Type aliases
    "Backend",
    # Functions
    "get_backend",
    "KernelProvider",
    "is_singleton",
]

from tsdm.utils.backends.numerical import (
    Backend,
    KernelProvider,
    get_backend,
    is_singleton,
)

# KERNELS = {
#     "where": where_kernel,
#     "clip": clip_kernel,
# }
