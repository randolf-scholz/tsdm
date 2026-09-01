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
    "kernels",
]


from . import generic, numpy, pandas, polars, pyarrow, torch  # ruff: ignore[I001]
from . import kernels  # must be imported last
from .kernels import *  # ruff: ignore[F403]

__all__ += kernels.__all__
