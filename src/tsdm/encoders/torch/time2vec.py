r"""Implementation of Time2Vec."""

__all__ = ["Time2VecEncoder", "Time2Vec"]

from dataclasses import KW_ONLY, dataclass, field
from typing import Final

import torch
from torch import Tensor, nn

from tsdm.constants import UNDEFINED
from tsdm.encoders.base import StaticEncoder
from tsdm.pprint import pprint_repr


class Time2Vec(nn.Module):
    r"""Learnable Time Encoding.

    References:
      - | Time2Vec: Learning a Vector Representation of Time
        | Seyed Mehran Kazemi, Rishab Goel, Sepehr Eghbali, Janahan Ramanan, Jaspreet
        | Sahota, Sanjay Thakur, Stella Wu, Cathal Smyth, Pascal Poupart, Marcus Brubaker
        | https://arxiv.org/abs/1907.05321
    """

    # Constants
    num_dim: Final[int]
    r"""Number of dimensions of the time encoding."""

    # Parameters
    freq: Tensor
    r"""Frequency of the time encoding."""
    phase: Tensor
    r"""Phase of the time encoding."""

    def __init__(self, *, num_dim: int, activation: str = "sin") -> None:
        super().__init__()
        self.num_dim = num_dim
        self.freq = nn.Parameter(torch.randn(num_dim - 1))
        self.phase = nn.Parameter(torch.randn(num_dim - 1))

        self.act = {
            "sin": torch.sin,
            "cos": torch.cos,
        }[activation]

    # Float[...] -> Float[..., D]
    def encode(self, t: Tensor) -> Tensor:
        z = torch.einsum("..., k -> ...k", t, self.freq) + self.phase
        z = self.act(z)
        return torch.cat([t.unsqueeze(dim=-1), z], dim=-1)

    # Float[..., D] -> Float[...]
    def decode(self, z: Tensor) -> Tensor:
        return z[..., 0]

    # Float[...] -> Float[..., D]
    def forward(self, t: Tensor) -> Tensor:
        return self.encode(t)

    # Float[..., D] -> Float[...]
    def inverse(self, z: Tensor) -> Tensor:
        return self.decode(z)


@pprint_repr
@dataclass(slots=True)
class Time2VecEncoder(StaticEncoder[Tensor, Tensor]):
    r"""Wraps Time2Vec encoder."""

    _: KW_ONLY

    # Constants
    num_dim: int
    r"""Number of dimensions of the time encoding."""
    activation: str
    r"""Activation function for the time encoding."""

    # Parameters
    freq: Tensor = field(init=False, default=UNDEFINED)
    r"""Frequency of the time encoding."""
    phase: Tensor = field(init=False, default=UNDEFINED)
    r"""Phase of the time encoding."""
    encoder: Time2Vec = field(init=False, default=UNDEFINED)
    r"""The wrapped encoder."""

    def __post_init__(self) -> None:
        self.encoder = Time2Vec(num_dim=self.num_dim, activation=self.activation)
        self.freq = self.encoder.freq
        self.phase = self.encoder.phase

    def encode(self, data: Tensor, /) -> Tensor:
        return self.encoder.encode(data)

    def decode(self, data: Tensor, /) -> Tensor:
        return self.encoder.decode(data)
