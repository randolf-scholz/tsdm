r"""PyTorch time-series sample structures and tensor-oriented utilities.

This backend exposes tensor representations of prediction samples, conversion
between supported layouts, and collation for variable-length batches. It also
reserves a namespace for PyTorch-native time-series dataset wrappers.
"""

__all__ = ["datasets", "samples"]

from . import datasets, samples
from .datasets import *  # ruff: ignore[F403]
from .samples import *  # ruff: ignore[F403]

__all__ += datasets.__all__
__all__ += samples.__all__
