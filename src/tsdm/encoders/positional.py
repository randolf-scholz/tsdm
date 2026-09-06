r"""Positional encoding for numpy."""

__all__ = ["PositionalEncoder"]

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from tsdm.pprint import pprint_repr
from tsdm.types import SupportsArrayUfunc

from .base import StaticEncoder


@pprint_repr
@dataclass(slots=True, init=False)
class PositionalEncoder[T: SupportsArrayUfunc](StaticEncoder[T, T]):
    r"""Positional encoding.

    .. math::
        x_{2 k}(t)   &:=\sin \left(\frac{t}{t^{2 k / τ}}\right) \\
        x_{2 k+1}(t) &:=\cos \left(\frac{t}{t^{2 k / τ}}\right)
    """

    # Constants
    num_dim: Final[int]
    r"""Number of dimensions."""

    # Buffers
    scale: Final[float]
    r"""Scale factor for positional encoding."""
    scales: Final[NDArray]
    r"""Scale factors for positional encoding."""

    def __init__(self, num_dim: int, scale: float) -> None:
        self.num_dim = num_dim
        self.scale = float(scale)
        self.scales = self.scale ** (-np.arange(0, num_dim + 2, 2) / num_dim)
        if self.scales[0] != 1.0:
            raise ValueError("Initial scale must be 1.0")

    # Float[...] -> Float[..., 2D]
    def encode[Arr: SupportsArrayUfunc](self, x: Arr, /) -> Arr:
        # Note: we simply concatenate the sin and cosine terms without interleaving them.
        z = np.einsum("..., d -> ...d", x, self.scales)
        return np.concatenate([np.sin(z), np.cos(z)], axis=-1)  # type: ignore

    # Float[..., 2D] -> Float[...]
    def decode[Arr: SupportsArrayUfunc](self, y: Arr, /) -> Arr:
        return np.arcsin(y[..., 0])  # type: ignore
