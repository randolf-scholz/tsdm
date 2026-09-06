r"""Positional Encoding in torch."""

__all__ = ["PositionalEncoding", "PositionalEncoder"]

from dataclasses import KW_ONLY, dataclass, field
from typing import Final

import torch
from torch import Tensor, nn

from tsdm.constants import UNDEFINED
from tsdm.encoders.base import StaticEncoder
from tsdm.pprint import pprint_repr


class PositionalEncoding(nn.Module):
    r"""Positional encoding.

    .. math::
        x_{2k}(t)   &≔\sin \left(\frac{t}{t^{2k/τ}}\right) \\
        x_{2k+1}(t) &≔\cos \left(\frac{t}{t^{2k/τ}}\right)
    """

    HP: dict = {
        "__name__": __qualname__,
        "__module__": __name__,
        "num_dim": int,
        "scale": float,
    }

    # Constants
    num_dim: Final[int]
    r"""Number of dimensions."""
    scale: Final[float]
    r"""Scale factor for positional encoding."""
    # Buffers
    scales: Tensor
    r"""Scale factors for positional encoding."""

    def __init__(self, *, num_dim: int, scale: float) -> None:
        super().__init__()
        self.num_dim = int(num_dim)
        self.scale = float(scale)
        self.register_buffer(
            "scales",
            self.scale ** (-2 * torch.arange(0, num_dim // 2) / (num_dim - 2)),
        )

        if self.num_dim % 2 != 0:
            raise ValueError("num_dim must be even")
        if self.scales[0] != 1.0:
            raise ValueError("Lowest scale must be 1.0")

    # Float[...] -> Float[..., 2D]
    def encode(self, t: Tensor, /) -> Tensor:
        z = torch.einsum("..., d -> ...d", t, self.scales)
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)

    # Float[..., 2D] -> Float[...]
    def decode(self, z: Tensor, /) -> Tensor:
        return torch.asin(z[..., 0])

    # Float[...] -> Float[..., 2D]
    def forward(self, t: Tensor, /) -> Tensor:
        return self.encode(t)

    # Float[..., 2D] -> Float[...]
    def inverse(self, t: Tensor, /) -> Tensor:
        return self.decode(t)


@pprint_repr
@dataclass(slots=True)
class PositionalEncoder(StaticEncoder[Tensor, Tensor]):
    r"""Wraps PositionalEncoder encoder."""

    _: KW_ONLY

    num_dim: int
    r"""Number of dimensions."""
    scale: float
    r"""Scale factor for positional encoding."""

    scales: Tensor = field(init=False, default=UNDEFINED)
    r"""Scale factors for positional encoding."""
    encoder: PositionalEncoding = field(init=False, default=UNDEFINED)
    r"""The wrapped encoder."""

    def __post_init__(self) -> None:
        self.encoder = PositionalEncoding(num_dim=self.num_dim, scale=self.scale)
        self.scales = self.encoder.scales

    def encode(self, data: Tensor, /) -> Tensor:
        return self.encoder.encode(data)

    def decode(self, data: Tensor, /) -> Tensor:
        return self.encoder.decode(data)
