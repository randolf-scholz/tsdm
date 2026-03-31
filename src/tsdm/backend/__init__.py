r"""Utilities for backends.

TODO: Consider using the python Array API for the backend interface.

Supports pandas / numpy / torch.
"""

__all__ = [
    # submodules
    "fallback",
    "generic",
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "torch",
    # Constants
    "BACKENDS",
    # Type aliases
    "BackendID",
    # Classes
    "Kernels",
    "Backend",
    # Functions
    "get_backend",
]


from . import fallback, generic, numpy, pandas, polars, pyarrow, torch
from .kernels import BACKENDS, Backend, BackendID, Kernels, get_backend
