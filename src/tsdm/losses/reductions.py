r"""Reduction methods for aggregating penalties."""

__all__ = ["Reduction", "apply_reduction"]

from collections.abc import Callable
from enum import StrEnum
from typing import assert_never

import torch
from torch import Tensor

from .base import Dim


class Reduction(StrEnum):
    r"""Reduction method for metrics."""

    SUM = "sum"
    MEAN = "mean"
    NONE = "none"

    def apply(self, values: Tensor, /, *, weight: Tensor | None = None) -> Tensor:
        r"""Apply an optional sample weight and outer reduction to losses."""
        if weight is not None:
            weight = torch.broadcast_to(weight, values.shape)
            values = weight * values

        match self:
            case Reduction.NONE:
                return values
            case Reduction.SUM:
                return values.sum()
            case Reduction.MEAN:
                return values.mean() if weight is None else values.sum() / weight.sum()
            case other:
                assert_never(other)


def apply_reduction(
    args: Tensor,
    /,
    *,
    inner_fun: Callable[..., Tensor],
    outer_fun: Callable[[Tensor], Tensor],
    dim: Dim = -1,
    mask: Tensor | None = None,  # Bool[..., *D]
    weight: Tensor | None = None,  # Float[..., *D] or Float[*D], non-negative
    scaled: bool = False,
) -> Tensor:
    r"""Apply a reduction to a tensor, optionally masked.

    .. math:: ℓ = ψ( agg_k ϕ(xₖ) )

    with inner function $ϕ$, outer function $ψ$ and aggregation function $agg_k$.
    Note that due to gradient accumulation, the arguments to $ϕ$ need to be masked
    beforehand.

    +------+--------+--------+-------------------------------------------+
    | mask | weight | scaled | formula                                   |
    +======+========+========+===========================================+
    | n    | n      | n      | $∑ₖxₖ$                                    |
    +------+--------+--------+-------------------------------------------+
    | n    | n      | y      | $∑ₖxₖ/K$                                  |
    +------+--------+--------+-------------------------------------------+
    | n    | y      | n      | $∑ₖwₖxₖ$                                  |
    +------+--------+--------+-------------------------------------------+
    | n    | y      | y      | $∑ₖwₖ/∑ⱼ⟦wⱼ≠0⟧⟦wₖ≠0 \? xₖ : 0⟧$           |
    +------+--------+--------+-------------------------------------------+
    | y    | n      | n      | $∑ₖ⟦mₖ \? xₖ : 0⟧$                        |
    +------+--------+--------+-------------------------------------------+
    | y    | n      | y      | $∑ₖ⟦mₖ \? xₖ/∑ⱼmⱼ : 0⟧$                   |
    +------+--------+--------+-------------------------------------------+
    | y    | y      | n      | $∑ₖwₖ⟦mₖ&(wₖ≠0) \? xₖ : 0⟧$               |
    +------+--------+--------+-------------------------------------------+
    | y    | y      | y      | $∑ₖwₖ/∑ⱼmⱼ&(wⱼ≠0)⟦mₖ&(wₖ≠0) \? xₖ : 0⟧$   |
    +------+--------+--------+-------------------------------------------+
    """
    raise NotImplementedError
