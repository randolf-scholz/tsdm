"""Utilities for backends.

Currently, supports pandas / numpy / torch.
"""

__all__ = [
    # Constants
    # "KERNELS",
    # Type aliases
    "BackendID",
    # Classes
    "Kernels",
    "KernelProvider",
    # Functions
    "get_backend",
    "is_singleton",
]

from tsdm.utils.backends.numerical import (
    BackendID,
    KernelProvider,
    Kernels,
    get_backend,
)
from tsdm.utils.backends.universal import is_singleton

# KERNELS = {
#     "where": where_kernel,
#     "clip": clip_kernel,
# }
