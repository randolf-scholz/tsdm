__all__ = ["quantile_error"]

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
