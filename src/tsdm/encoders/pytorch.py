r"""Encoders that work with torch tensors."""

__all__ = [
    # Classes
    "Time2Vec",
    "PositionalEncoding",
    # Encoders
    "PositionalEncoder",
    "TensorEncoder",
    "Time2VecEncoder",
]

import torch
from torch import Tensor, jit, nn
from typing_extensions import Final

from tsdm.encoders.base import BaseEncoder
from tsdm.utils.decorators import autojit

# TODO: Add TensorEncoder


class TensorEncoder(BaseEncoder):
    r"""Encodes nested data as tensors."""

    def encode(self, data):
        match data:
            case Tensor():
                return data
            case list():
                return [self.encode(d) for d in data]
            case tuple():
                return tuple(self.encode(d) for d in data)
            case dict():
                return {k: self.encode(v) for k, v in data.items()}
            case set():
                return {self.encode(d) for d in data}
            case frozenset():
                return frozenset({self.encode(d) for d in data})
            case _:
                try:
                    return torch.tensor(data)
                except Exception as exc:
                    raise TypeError(f"Cannot encode data of type {type(data)}") from exc

    def decode(self, data):
        return data.numpy()


@autojit
class Time2Vec(nn.Module):
    r"""Learnable Time Encoding.

    References:
      - | Time2Vec: Learning a Vector Representation of Time
        | Seyed Mehran Kazemi, Rishab Goel, Sepehr Eghbali, Janahan Ramanan, Jaspreet
        | Sahota, Sanjay Thakur, Stella Wu, Cathal Smyth, Pascal Poupart, Marcus Brubaker
        | https: // arxiv.org / abs / 1907.05321
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

    @jit.export
    def encode(self, t: Tensor) -> Tensor:
        r""".. signature:: ``(..., d) -> ...``."""
        z = torch.einsum("..., k -> ...k", t, self.freq) + self.phase
        z = self.act(z)
        return torch.cat([t.unsqueeze(dim=-1), z], dim=-1)

    @jit.export
    def decode(self, z: Tensor) -> Tensor:
        r""".. signature:: ``(..., d) -> ...``."""
        return z[..., 0]

    @jit.export
    def forward(self, t: Tensor) -> Tensor:
        r""".. signature:: ``... -> (..., d)``."""
        return self.encode(t)

    @jit.export
    def inverse(self, z: Tensor) -> Tensor:
        r""".. signature:: ``(..., d) -> ...``."""
        return self.decode(z)


@autojit
class PositionalEncoding(nn.Module):
    r"""Positional encoding.

    .. math::
        x_{2k}(t)   &≔\sin \left(\frac{t}{t^{2k/τ}}\right) \\
        x_{2k+1}(t) &≔\cos \left(\frac{t}{t^{2k/τ}}\right)
    """

    HP: dict = {
        "__name__": __qualname__,
        "__doc__": __doc__,
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
        assert num_dim % 2 == 0, "num_dim must be even"
        self.num_dim = num_dim
        self.scale = float(scale)
        scales = self.scale ** (-2 * torch.arange(0, num_dim // 2) / (num_dim - 2))
        assert scales[0] == 1.0, "Something went wrong."
        self.register_buffer("scales", scales)

    @jit.export
    def encode(self, t: Tensor) -> Tensor:
        r""".. signature:: ``(..., d) -> ...``."""
        z = torch.einsum("..., d -> ...d", t, self.scales)
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)

    @jit.export
    def decode(self, z: Tensor) -> Tensor:
        r""".. signature:: ``(..., 2d) -> ...``."""
        return torch.asin(z[..., 0])

    @jit.export
    def forward(self, t: Tensor) -> Tensor:
        r""".. signature:: ``... -> (..., 2d)``.

        Note: we simply concatenate the sin and cosine terms without interleaving them.
        """
        return self.encode(t)

    @jit.export
    def inverse(self, t: Tensor) -> Tensor:
        r""".. signature:: ``(..., 2d) -> ...``."""
        return self.decode(t)


class Time2VecEncoder(BaseEncoder):
    r"""Wraps Time2Vec encoder."""

    def __init__(self, *, num_dim: int, activation: str = "sin") -> None:
        self.encoder = Time2Vec(num_dim=num_dim, activation=activation)

    def encode(self, data: Tensor, /) -> Tensor:
        return self.encoder.encode(data)

    def decode(self, data: Tensor, /) -> Tensor:
        return self.encoder.decode(data)


class PositionalEncoder(BaseEncoder):
    r"""Wraps PositionalEncoder encoder."""

    def __init__(self, *, num_dim: int, scale: float) -> None:
        self.encoder = PositionalEncoding(num_dim=num_dim, scale=scale)

    def encode(self, data: Tensor, /) -> Tensor:
        return self.encoder.encode(data)

    def decode(self, data: Tensor, /) -> Tensor:
        return self.encoder.decode(data)
