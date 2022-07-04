"""TODO: add package summary.

TODO: add package description.
"""

__all__ = [
    # Functions
    "cumulative_and",
    "cumulative_or",
    "aggregate_and",
    "aggregate_or",
]

from typing import Union

import torch
from torch import Tensor, jit


@jit.script
def cumulative_and(x: Tensor, dim: Union[None, int] = None) -> Tensor:
    r"""Cumulative aggregation with logical ``AND`` $yᵢ = ⋀_{j≤i} xⱼ$."""
    # TODO: rewrite with enumerate and &= when BUGS are fixed
    # BUG: https://github.com/pytorch/pytorch/issues/67142
    # BUG: https://github.com/pytorch/pytorch/issues/68064
    dim = 0 if dim is None else dim
    y = x.clone().swapaxes(0, dim)
    for i in range(1, len(y)):
        y[i] = y[i] & y[i - 1]
    return y.swapaxes(0, dim)


@jit.script
def cumulative_or(x: Tensor, dim: Union[None, int] = None) -> Tensor:
    r"""Cumulative aggregation with logical ``OR`` $yᵢ = ⋁_{j≤i} xⱼ$."""
    # TODO: rewrite with enumerate and &= when BUGS are fixed
    # BUG: https://github.com/pytorch/pytorch/issues/67142
    # BUG: https://github.com/pytorch/pytorch/issues/68064
    dim = 0 if dim is None else dim
    y = x.clone().swapaxes(0, dim)
    for i in range(1, len(y)):
        y[i] = y[i] | y[i - 1]
    return y.swapaxes(0, dim)


@jit.script
def aggregate_and(
    x: Tensor,
    dim: Union[None, int, list[int]] = None,
    keepdim: bool = False,
) -> Tensor:
    r"""Compute logical ``AND`` across dim."""
    if dim is None:
        dims = list(range(x.ndim))
    elif isinstance(dim, int):
        dims = [dim]
    else:
        dims = dim

    if isinstance(dims, Tensor):
        if keepdim:
            for d in dims:
                x = torch.all(x, dim=d, keepdim=keepdim)
        else:
            for i, d in enumerate(dims):
                x = torch.all(x, dim=d - i, keepdim=keepdim)

    if keepdim:
        for d in dims:
            x = torch.all(x, dim=d, keepdim=keepdim)
    else:
        for i, d in enumerate(dims):
            x = torch.all(x, dim=d - i, keepdim=keepdim)

    return x


@jit.script
def aggregate_or(
    x: Tensor,
    dim: Union[None, int, list[int]] = None,
    keepdim: bool = False,
) -> Tensor:
    r"""Compute logical ``OR`` across dim."""
    if dim is None:
        dims = list(range(x.ndim))
    elif isinstance(dim, int):
        dims = [dim]
    else:
        dims = dim

    if keepdim:
        for d in dims:
            x = torch.any(x, dim=d, keepdim=keepdim)
    else:
        for i, d in enumerate(dims):
            x = torch.any(x, dim=d - i, keepdim=keepdim)

    return x
