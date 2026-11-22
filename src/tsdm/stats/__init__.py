r"""Statistical distances and summaries."""

__all__ = [
    "distances",
]

from . import distances
from .distances import *  # ruff: ignore[F403]

__all__ += distances.__all__
