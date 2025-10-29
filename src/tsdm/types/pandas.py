r"""Type aliases specific to pandas."""

__all__ = ["MaybeNA"]

from pandas.api.typing import NAType

type MaybeNA[T] = T | NAType
r"""Type Alias for nullable types (pandas-specific)."""
