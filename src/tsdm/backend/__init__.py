r"""Utilities for backends.

TODO: Consider using the python Array API for the backend interface.

Supports pandas / numpy / torch.
"""

__all__ = [
    # submodules
    "generic",
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "torch",
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


from tsdm.backend import generic, numpy, pandas, polars, pyarrow, torch
from tsdm.backend.kernels import Backend, BackendID, Kernels, get_backend
