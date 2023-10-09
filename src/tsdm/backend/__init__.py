"""Utilities for backends.

Supports pandas / numpy / torch.
"""

__all__ = [
    # Constants
    # "KERNELS",
    # Type aliases
    "BackendID",
    # Classes
    "Kernels",
    "Backend",
    # Functions
    "get_backend",
    "is_singleton",
]

from tsdm.backend._backend import Backend, BackendID, Kernels, get_backend
from tsdm.backend.universal import is_singleton

# KERNELS = {
#     "where": where_kernel,
#     "clip": clip_kernel,
# }
