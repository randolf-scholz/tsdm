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
]

from tsdm.utils.backends.numerical import Backend, KernelProvider, get_backend

# KERNELS = {
#     "where": where_kernel,
#     "clip": clip_kernel,
# }
