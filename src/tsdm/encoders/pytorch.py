r"""Encoders that work with torch tensors."""

__all__ = [
    # Classes
    "Time2Vec",
    "PositionalEncoding",
    # Encoders
    "PositionalEncoder",
    "RecursiveTensorEncoder",
    "Time2VecEncoder",
]

from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Final

import torch
from numpy.typing import NDArray
from torch import Tensor, jit, nn

from tsdm.encoders.base import BaseEncoder
from tsdm.types.aliases import NestedBuiltin
from tsdm.utils.decorators import autojit, pprint_repr


@autojit
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
        r""".. signature:: ``... -> (..., 2d)``."""
        return self.encode(t)

    @jit.export
    def inverse(self, t: Tensor) -> Tensor:
        r""".. signature:: ``(..., 2d) -> ...``."""
        return self.decode(t)


class RecursiveTensorEncoder(
    BaseEncoder[NestedBuiltin[NDArray], NestedBuiltin[Tensor]]
):
    r"""Encodes nested data as tensors."""

    @property
    def params(self) -> dict[str, Any]:
        return {}

    def _encode_impl(self, data: NestedBuiltin[NDArray], /) -> NestedBuiltin[Tensor]:
        match data:
            case list(seq):
                return [self.encode(d) for d in seq]
            case tuple(tup):
                return tuple(self.encode(d) for d in tup)
            case dict(mapping):
                return {k: self.encode(v) for k, v in mapping.items()}
            case set(items):
                return {self.encode(d) for d in items}  # pyright: ignore[reportUnhashable]
            case frozenset(items):
                return frozenset({self.encode(d) for d in items})  # pyright: ignore[reportUnhashable]
            case array:
                try:
                    return torch.tensor(array)
                except Exception as exc:
                    raise TypeError(f"Cannot encode data of type {type(data)}") from exc

    def _decode_impl(self, data: NestedBuiltin[Tensor], /) -> NestedBuiltin[NDArray]:
        match data:
            case list(seq):
                return [self.decode(d) for d in seq]
            case tuple(tup):
                return tuple(self.decode(d) for d in tup)
            case dict(mapping):
                return {k: self.decode(v) for k, v in mapping.items()}
            case set(items):
                return {self.decode(d) for d in items}  # pyright: ignore[reportUnhashable]
            case frozenset(items):
                return frozenset({self.decode(d) for d in items})  # pyright: ignore[reportUnhashable]
            case tensor:
                try:
                    return tensor.numpy()
                except Exception as exc:
                    raise TypeError(f"Cannot encode data of type {type(data)}") from exc


@pprint_repr
@dataclass
class Time2VecEncoder(BaseEncoder[Tensor, Tensor]):
    r"""Wraps Time2Vec encoder."""

    # Constants
    num_dim: int
    r"""Number of dimensions of the time encoding."""
    activation: str
    r"""Activation function for the time encoding."""
    # Parameters
    freq: Tensor = field(init=False)
    r"""Frequency of the time encoding."""
    phase: Tensor = field(init=False)
    r"""Phase of the time encoding."""
    encoder: Time2Vec = field(init=False)
    r"""The wrapped encoder."""

    def __post_init__(self):
        self.encoder = Time2Vec(num_dim=self.num_dim, activation=self.activation)
        self.freq = self.encoder.freq
        self.phase = self.encoder.phase

    def _encode_impl(self, data: Tensor, /) -> Tensor:
        return self.encoder.encode(data)

    def _decode_impl(self, data: Tensor, /) -> Tensor:
        return self.encoder.decode(data)


@pprint_repr
@dataclass
class PositionalEncoder(BaseEncoder[Tensor, Tensor]):
    r"""Wraps PositionalEncoder encoder."""

    _: KW_ONLY
    # Constants
    num_dim: int
    r"""Number of dimensions."""
    scale: float
    r"""Scale factor for positional encoding."""
    scales: Tensor = field(init=False)
    r"""Scale factors for positional encoding."""
    encoder: PositionalEncoding = field(init=False)
    r"""The wrapped encoder."""

    def __post_init__(self):
        self.encoder = PositionalEncoding(num_dim=self.num_dim, scale=self.scale)
        self.scales = self.encoder.scales

    def _encode_impl(self, data: Tensor, /) -> Tensor:
        return self.encoder.encode(data)

    def _decode_impl(self, data: Tensor, /) -> Tensor:
        return self.encoder.decode(data)
