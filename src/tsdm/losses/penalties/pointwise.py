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
    r"""Return the element-wise pinball penalty for a residual tensor.

    For the residual $r=x̂-x$, the q-quantile errors is:

    $$ ρ_q(r) ≔ ⟦r ≤ 0 \? q⋅|r| : (1-q)⋅|r| ⟧ = \max( (q-1)r, qr) $$

    Raises:
        ValueError: If $q∉[0,1]$.

    References:
        - Deep State Space Models for Time Series Forecasting <br>
          Syama Sundar Rangapuram et al. <br>
          Advances in Neural Information Processing Systems 31 (NeurIPS 2018) <br>
          <https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html>
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
    r"""Compute a masked, weighted entry-wise norm along `dim`.

    Without masks or weights, the ordinary and scaled forms are

    $$\begin{aligned}
        ‖x‖ₚ &≔ \Bigl(∑ₖ |xₖ|ᵖ\Bigr)^{1/p} \\
        ‖x‖_{p⁎} &≔ \Bigl(K⁻¹∑ₖ |xₖ|ᵖ\Bigr)^{1/p}
    \end{aligned}$$

    For masks and weights, let $A$ contain entries selected by the mask whose
    weights are nonzero. Omitted masks select all entries; omitted weights are one.
    Then the general forms are

    $$\begin{aligned}
        ‖x‖_{m,w,p} &≔ \Bigl(∑_{k∈A} wₖ|xₖ|ᵖ\Bigr)^{1/p} \\
        ‖x‖_{m,w,p⁎} &≔ \Bigl(|A|⁻¹∑_{k∈A} wₖ|xₖ|ᵖ\Bigr)^{1/p}
    \end{aligned}$$

    Inactive values are selected out before taking powers, so masked NaNs and
    infinities do not contribute. Weights should be nonnegative, broadcast to the
    data shape, and have one axis per dimension in `dim`. Scaling counts active
    entries rather than summing their weights. A scaled reduction with no active
    entries raises `ValueError`.

    Finite nonzero positive and negative orders are supported. Orders zero and
    positive or negative infinity raise `NotImplementedError`; NaN raises
    `ValueError`. An empty `dim` tuple performs no inner sum.
    """
    dims = (dim,) if isinstance(dim, int) else dim

    if weight is not None and weight.ndim != len(dims):
        raise ValueError(f"Expected {weight.ndim=} to equal {len(dims)=}")

    # fmt: off
    active: Tensor | None
    match mask, weight:
        case None, None: active = None
        case _,    None: active = mask
        case None, _   : active = weight != 0  # ty: ignore[invalid-assignment]
        case _,    _   : active = mask & (weight != 0)  # ty: ignore[unsupported-operator]
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
