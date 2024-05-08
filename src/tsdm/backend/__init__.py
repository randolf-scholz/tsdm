r"""Utilities for backends.

Supports pandas / numpy / torch.
"""

__all__ = [
    # submodules
    "numpy",
    "pandas",
    "pyarrow",
    "torch",
    "generic",
    # Constants
    # "KERNELS",
    # Type aliases
    "BackendID",
    # Classes
    "Kernels",
    "Backend",
    # Functions
    "get_backend",
]


from tsdm.backend import generic, numpy, pandas, pyarrow, torch
from tsdm.backend.kernels import Backend, BackendID, Kernels, get_backend
