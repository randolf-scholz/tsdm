r"""Positional encoding for numpy."""

__all__ = ["PositionalEncoder"]

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from tsdm.types import SupportsArrayUfunc
from tsdm.utils import pretty_repr

from .base import StaticEncoder


@pretty_repr
@dataclass(slots=True, init=False)
class PositionalEncoder[T: SupportsArrayUfunc](StaticEncoder[T, T]):
    r"""Encode time using sine and cosine components.

    $$\begin{aligned}
        sₖ &= τ^{-2k/d} \\
        z(t) &= (\sin(s₀t), …, \sin(sₘ₋₁t), \cos(s₀t), …, \cos(sₘ₋₁t))
    \end{aligned}$$

    Here $d$ is `num_dim`, $τ$ is `scale`, and $m$ is the length of
    `arange(0, num_dim + 2, 2)`. Use a positive dimension and scale.
    The output has $2m$ components along its last axis.
    Decoding uses the arcsine of the first component, so it recovers the input
    only on $[-π/2, π/2]$.
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
