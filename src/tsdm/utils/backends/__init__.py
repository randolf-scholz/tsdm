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
    "true_like",
    "false_like",
]

from tsdm.utils.backends.numerical import (
    Backend,
    KernelProvider,
    false_like,
    get_backend,
    is_singleton,
    true_like,
)

# KERNELS = {
#     "where": where_kernel,
#     "clip": clip_kernel,
# }
