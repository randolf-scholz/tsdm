r"""Numerical kernels for NumPy, pandas, Polars, PyArrow, and PyTorch.

Todo:
    Consider using the Python Array API for the backend interface.
"""

__all__ = [
    # submodules
    "generic",
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "torch",
    "util",
    "kernels",
]


from . import generic, numpy, pandas, polars, pyarrow, torch, util  # ruff: ignore[I001]
from . import kernels  # must be imported last
from .kernels import *  # ruff: ignore[F403]

__all__ += kernels.__all__
