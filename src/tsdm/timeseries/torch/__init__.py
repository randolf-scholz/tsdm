r"""PyTorch time-series sample structures and tensor-oriented utilities.

This backend exposes tensor representations of prediction samples and
collation for variable-length batches. Conversion between layouts and
PyTorch-native time-series dataset wrappers are reserved for future implementation.
"""

__all__ = ["datasets", "samples"]

from . import datasets, samples
from .datasets import *  # ruff: ignore[F403]
from .samples import *  # ruff: ignore[F403]

__all__ += datasets.__all__
__all__ += samples.__all__
