r"""Cumulative logical functions."""

__all__ = [
    # Functions
    "cumulative_and",
    "cumulative_or",
    "cumulative_xor",
]

from torch import Tensor, jit


@jit.script
def cumulative_and(x: Tensor, dim: int = 0) -> Tensor:
    r"""Cumulative aggregation with logical ``AND`` $yᵢ = ⋀_{j≤i} xⱼ$."""
    y = x.clone().swapaxes(0, dim)
    for i in range(1, len(y)):
        y[i] = y[i] & y[i - 1]
    return y.swapaxes(0, dim)


@jit.script
def cumulative_or(x: Tensor, dim: int = 0) -> Tensor:
    r"""Cumulative aggregation with logical ``OR`` $yᵢ = ⋁_{j≤i} xⱼ$."""
    y = x.clone().swapaxes(0, dim)
    for i in range(1, len(y)):
        y[i] = y[i] | y[i - 1]
    return y.swapaxes(0, dim)


@jit.script
def cumulative_xor(x: Tensor, dim: int = 0) -> Tensor:
    r"""Cumulative aggregation with logical ``XOR`` $yᵢ = ⊕_{j≤i} xⱼ$."""
    y = x.clone().swapaxes(0, dim)
    for i in range(1, len(y)):
        y[i] = y[i] ^ y[i - 1]
    return y.swapaxes(0, dim)
