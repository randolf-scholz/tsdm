r"""Cumulative logical functions."""

__all__ = [
    # Functions
    "aggregate_and",
    "aggregate_or",
    "cumulative_and",
    "cumulative_or",
]

import torch
from torch import Tensor, jit
from typing_extensions import Optional


@jit.script
def aggregate_and(
    x: Tensor,
    dim: None | int | list[int] = None,
    keepdim: bool = False,
) -> Tensor:
    r"""Compute logical ``AND`` across dim."""
    dims: list[int] = (
        list(range(x.ndim))
        if dim is None
        else [dim]
        if isinstance(dim, int)
        else list(dim)
    )

    if keepdim:
        for d in dims:
            x = torch.all(x, dim=d, keepdim=keepdim)
        return x

    for i, d in enumerate(dims):
        x = torch.all(x, dim=d - i, keepdim=keepdim)
    return x


@jit.script
def aggregate_or(
    x: Tensor,
    dim: None | int | list[int] = None,
    keepdim: bool = False,
) -> Tensor:
    r"""Compute logical ``OR`` across dim."""
    dims: list[int] = (
        list(range(x.ndim))
        if dim is None
        else [dim]
        if isinstance(dim, int)
        else list(dim)
    )

    if keepdim:
        for d in dims:
            x = torch.any(x, dim=d, keepdim=keepdim)
        return x

    for i, d in enumerate(dims):
        x = torch.any(x, dim=d - i, keepdim=keepdim)
    return x


@jit.script
def cumulative_and(x: Tensor, dim: Optional[int] = None) -> Tensor:
    r"""Cumulative aggregation with logical ``AND`` $yᵢ = ⋀_{j≤i} xⱼ$."""
    dim = 0 if dim is None else dim
    y = x.clone().swapaxes(0, dim)
    for i in range(1, len(y)):
        y[i] = y[i] & y[i - 1]
    return y.swapaxes(0, dim)


@jit.script
def cumulative_or(x: Tensor, dim: Optional[int] = None) -> Tensor:
    r"""Cumulative aggregation with logical ``OR`` $yᵢ = ⋁_{j≤i} xⱼ$."""
    dim = 0 if dim is None else dim
    y = x.clone().swapaxes(0, dim)
    for i in range(1, len(y)):
        y[i] = y[i] | y[i - 1]
    return y.swapaxes(0, dim)
