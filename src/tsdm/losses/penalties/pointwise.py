r"""Penalty functions for vector data."""

__all__ = [
    "quantile_error",
    "lp_norm",
]


import torch
from torch import Tensor


def quantile_error(
    r,  # Float[...]
    /,
    *,
    q: float = 0.5,
) -> Tensor:  # Float[...]
    r"""Return the q-quantile / pinball loss.

    For a residual scalar $r=x̂-x$, this is just:

    .. math:: ρ_q(r) ≔ ⟦r ≤ 0 \? q⋅|r| : (1-q)⋅|r| ⟧ = \max( (q-1)r, qr)

    References:
        - | Deep State Space Models for Time Series Forecasting
          | Syama Sundar Rangapuram et al.
          | Advances in Neural Information Processing Systems 31 (NeurIPS 2018)
          | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"q must be in [0, 1], but got {q}")
    # simplified formula
    return torch.maximum((q - 1) * r, q * r)


def lp_norm(
    x: Tensor,  # Float[..., *D]
    /,
    *,
    dim: int | tuple[int, ...],  # *D
    p: float = 2.0,
    mask: Tensor | None = None,  # Bool[..., *D]
    weight: Tensor | None = None,  # Float[..., *D] or Float[*D], non-negative
    scaled: bool = False,
) -> Tensor:  # Float[...]
    r"""Compute (possibly scaled) Lₚ-norm.

    .. math:: ‖x‖ₚ     ≔ \sqrt[p]{∑ₖ|xₖ|ᵖ}
    .. math:: ‖x‖_{p⁎} ≔ \sqrt[p]{(1/K)∑ₖ|xₖ|ᵖ}

    And masked versions:

    .. math:: ‖x‖_{m,p}  ≔ \sqrt[p]{∑ₖ ⟦mₖ \? |xₖ|ᵖ : 0⟧ }
    .. math:: ‖x‖_{m,p⁎} ≔ \sqrt[p]{(1/∑ⱼmⱼ) ∑ₖ⟦mₖ \? |xₖ|ᵖ : 0⟧ }

    Moreover, one can introduce channel weights:

    .. math:: ‖x‖_{w,p}  ≔ \sqrt[p]{∑ₖwₖ|xₖ|ᵖ}
    .. math:: ‖x‖_{w,p⁎} ≔ \sqrt[p]{(1/∑ⱼ⟦wⱼ≠0⟧)∑ₖwₖ|xₖ|ᵖ}

    and with both weights and masks:

    .. math:: ‖x‖_{m,w,p}  ≔ \sqrt[p]{∑ₖ wₖ⟦mₖ&(wₖ≠0) \? |xₖ|ᵖ : 0⟧ }
    .. math:: ‖x‖_{m,w,p⁎} ≔ \sqrt[p]{(1/∑ⱼmⱼ&(wⱼ≠0)) ∑ₖwₖ⟦mₖ&(wₖ≠0) \? |xₖ|ᵖ : 0⟧ }

    Note:
        For weighted scaled norms, zero-weight entries are treated as masked:
        the denominator is ``sum(weight != 0)`` without a mask,
        and ``sum(mask & (weight != 0))`` with one.

        An empty ``dim`` tuple performs no inner reduction.
    """
    dims = (dim,) if isinstance(dim, int) else dim

    if weight is not None and weight.ndim != len(dims):
        raise ValueError(f"Expected {weight.ndim=} to equal {len(dims)=}")

    # fmt: off
    active: Tensor | None
    match mask, weight:
        case None, None: active = None
        case _,    None: active = mask
        case None, _   : active = weight != 0  # type: ignore
        case _,    _   : active = mask & (weight != 0)  # type: ignore
    # fmt: on

    if p != p:
        raise ValueError("p must not be NaN.")
    if p == torch.inf:
        raise NotImplementedError
    if p == -torch.inf:
        raise NotImplementedError
    if p == 0.0:
        # TODO: if scaled, use geometric mean.
        raise NotImplementedError
    if p > 0:  # p∈(0,∞)
        if active is not None:
            x = torch.where(active, x, 0.0)
        s = x.abs().pow(p)
    else:  # p∈(-∞,0)
        if active is not None:
            x = torch.where(active, x, 1.0)
        s = x.abs().pow(p)
        if active is not None:
            s = torch.where(active, s, 0.0)

    if weight is not None:
        s = weight * s

    r = torch.sum(s, dim=dims) if dims else s

    if scaled:
        numel = (
            torch.Size(x.shape[d] for d in dims).numel()
            if active is None
            else active.sum(dim=dims)
        )

        if torch.any(torch.as_tensor(numel) == 0):
            raise ValueError("Cannot scale a norm with no unmasked elements.")
        r = r.div(numel)

    result = r.pow(1 / p)
    assert result.ndim == x.ndim - len(dims)
    return result
