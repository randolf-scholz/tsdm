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
    "kernels",
]


from . import generic, kernels, numpy, pandas, polars, pyarrow, torch
from .kernels import *  # ruff: ignore[F403]

__all__ += kernels.__all__
