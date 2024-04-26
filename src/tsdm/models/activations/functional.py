r"""Implementations of activation functions.

Notes:
    Contains activations in functional form.
    - See `tsdm.models.activations.modular` for modular implementations.
"""

__all__ = [
    # Functions
    "geglu",
    "hard_bend",
    "reglu",
]

import torch
from torch import Tensor, jit, nn


@jit.script
def reglu(x: Tensor) -> Tensor:
    r"""Regularized gelu activation function."""
    a, b = x.chunk(2, dim=-1)
    return a * nn.functional.relu(b)


@jit.script
def geglu(x: Tensor) -> Tensor:
    r"""Gelu activation function."""
    a, b = x.chunk(2, dim=-1)
    return a * nn.functional.gelu(b)


@jit.script
def hard_bend(x: Tensor, a: float = 1, t: float = 1) -> Tensor:
    r"""Hard step activation function."""
    exp_at = torch.tensor(a * t, device=x.device, dtype=x.dtype).exp()
    mask = x.abs() <= t / (exp_at - 1)
    return torch.where(mask, exp_at * x, x + torch.sign(x) * t)
