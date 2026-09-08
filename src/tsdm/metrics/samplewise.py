r"""Implementations of metrics for (non-sequential) data.

These metrics support missing values through the mask argument.
"""

__all__ = [
    "RMSE_Loss",
    "MSE_Loss",
    "MAE_loss",
    "LP_Loss",
    "Q_Quantile",
    "Reduction",
    # functions
    "rmse_loss",
    "mse_loss",
    "mae_loss",
    "lp_loss",
    "lp_norm",
    "q_quantile",
    "apply_reduction",
]

from collections.abc import Callable
from enum import StrEnum
from typing import Any, Final

import torch
from torch import Tensor, nn

from .base import BaseMetric

type Dim = int | tuple[int, ...] | None


class Reduction(StrEnum):
    r"""Reduction method for metrics."""

    SUM = "sum"
    MEAN = "mean"
    NONE = "none"


UNDEFINED: Final[Any] = object()


def apply_reduction(
    args: Tensor,
    /,
    *,
    inner_fun: Callable[..., Tensor],
    outer_fun: Callable[[Tensor], Tensor],
    dim: Dim = UNDEFINED,
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


def lp_norm(
    x: Tensor,  # Float[..., *D]
    /,
    *,
    p: float = 2.0,
    dim: Dim = UNDEFINED,  # *D
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
    dims = tuple(
        ((-1,) if weight is None else range(-weight.ndim, 0))
        if dim is UNDEFINED
        else (range(x.ndim) if dim is None else (dim,) if isinstance(dim, int) else dim)
    )
    if weight is not None and weight.ndim != len(dims):
        raise ValueError(f"Expected {weight.ndim=} to equal {len(dims)=}")

    # fmt: off
    active: Tensor | None
    match mask, weight:
        case None, None: active = None
        case _,    None: active = mask
        case None, _   : active = weight != 0  # pyrefly: ignore[bad-assignment]
        case _,    _   : active = mask & (weight != 0)  # pyrefly: ignore[unsupported-operation]
    # fmt: on

    if p == torch.inf:  # noqa: SIM114
        raise NotImplementedError
    elif p == -torch.inf:  # noqa: RET506
        raise NotImplementedError
    elif p == 0.0:
        # TODO: if scaled, use geometric mean.
        raise NotImplementedError
    elif p > 0:
        if active is not None:
            x = torch.where(active, x, 0.0)
        s = x.abs().pow(p)
    else:  # p<0
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

    return r.pow(1 / p)


def _reduce_loss(
    losses: Tensor,
    /,
    *,
    weight: Tensor | None = None,
    reduction: Reduction = Reduction.MEAN,
) -> Tensor:
    r"""Apply an optional sample weight and outer reduction to losses."""
    if weight is not None:
        weight = torch.broadcast_to(weight, losses.shape)
        losses = weight * losses

    match reduction:
        case Reduction.SUM:
            return losses.sum()
        case Reduction.MEAN:
            return losses.mean() if weight is None else losses.sum() / weight.sum()
        case Reduction.NONE:
            return losses
        case _:
            raise ValueError(f"Invalid reduction: {reduction}")


def lp_loss(
    *,
    predictions: Tensor,  # Float[..., *D]
    targets: Tensor,  # Float[..., *D],
    p: float = 2.0,
    dim: Dim = -1,  # *D
    mask: Tensor | None = None,  # Bool[..., *D],
    weight: Tensor | None = None,  # Float[...], sample weights
    channel_weight: Tensor | None = None,  # Float[..., *D], channel weights
    reduction: Reduction = Reduction.MEAN,
    scaled: bool = False,
) -> Tensor:  # Float[()]
    r"""Compute the sample-weighted $p$-norm loss.

    .. math:: ℓₚ(x̂，x) ≔ 𝔼[‖x̂ - x‖ₚ]

    ``channel_weight`` is applied within the $p$-norm along ``dim``. ``weight``
    is applied to the resulting per-sample losses before ``reduction``.
    """
    norms = lp_norm(
        predictions - targets,
        p=p,
        dim=dim,
        mask=mask,
        weight=channel_weight,
        scaled=scaled,
    )
    return _reduce_loss(norms, weight=weight, reduction=reduction)


def mae_loss(
    *,
    predictions: Tensor,  # Float[..., *D]
    targets: Tensor,  # Float[..., *D]
    dim: Dim = -1,
    mask: Tensor | None = None,  # Bool[..., *D],
    weight: Tensor | None = None,  # Float[...], sample weights
    channel_weight: Tensor | None = None,  # Float[..., *D], channel weights
    reduction: Reduction = Reduction.MEAN,
    scaled: bool = False,
) -> Tensor:  # Float[()]
    r"""Compute the sample-weighted mean absolute error.

    .. math:: mae(x̂，x) ≔ 𝔼[‖x̂ - x‖₂]
    """
    return lp_loss(
        predictions=predictions,
        targets=targets,
        p=1.0,
        dim=dim,
        mask=mask,
        weight=weight,
        channel_weight=channel_weight,
        scaled=scaled,
        reduction=reduction,
    )


def mse_loss(
    *,
    predictions: Tensor,  # Float[..., *D]
    targets: Tensor,  # Float[..., *D]
    dim: Dim = -1,
    mask: Tensor | None = None,  # Bool[..., *D],
    weight: Tensor | None = None,  # Float[...], sample weights
    channel_weight: Tensor | None = None,  # Float[..., *D], channel weights
    scaled: bool = False,
    reduction: Reduction = Reduction.MEAN,
) -> Tensor:  # Float[()]
    r"""Compute the sample-weighted mean squared error.

    .. math:: mse(x̂，x) ≔ 𝔼[‖x̂ - x‖₂²]
    """
    losses = lp_loss(
        predictions=predictions,
        targets=targets,
        p=2.0,
        dim=dim,
        mask=mask,
        channel_weight=channel_weight,
        scaled=scaled,
        reduction=Reduction.NONE,
    ).square()
    return _reduce_loss(losses, weight=weight, reduction=reduction)


def rmse_loss(
    *,
    predictions: Tensor,  # Float[..., *D]
    targets: Tensor,  # Float[..., *D]
    dim: Dim = -1,
    mask: Tensor | None = None,  # Bool[..., *D],
    weight: Tensor | None = None,  # Float[...], sample weights
    channel_weight: Tensor | None = None,  # Float[..., *D], channel weights
    reduction: Reduction = Reduction.MEAN,
    scaled: bool = False,
) -> Tensor:  # Float[()]
    r"""Compute the sample-weighted root mean squared error.

    .. math:: 𝗋𝗆𝗌𝖾(x̂，x) ≔ \sqrt{ 𝔼[‖x̂ - x‖₂²] }
    """
    squared_norms = lp_loss(
        predictions=predictions,
        targets=targets,
        p=2.0,
        dim=dim,
        mask=mask,
        channel_weight=channel_weight,
        scaled=scaled,
        reduction=Reduction.NONE,
    ).square()
    return _reduce_loss(squared_norms, weight=weight, reduction=reduction).sqrt()


class LP_Loss(BaseMetric):
    r"""$Lᵖ$ Loss.

    Given two random vectors $x̂,x∈ℝᴷ$, the $Lᵖ$-loss is defined as:

    .. math:: 𝖱𝖬𝖲𝖤(x̂，x) ≔ \sqrt[p]{𝔼[‖x̂ - x‖ᵖ]}

    Given $N$ random samples $x_1, …, x_N$ and $x̂_1, …, x̂_N$, it can be estimated as:

    .. math:: 𝖱𝖬𝖲𝖤(x̂，x) ∼ \sqrt[p]{\frac{1}{N}∑ₙ₌₁ᴺ ‖x̂ₙ - xₙ‖ᵖ}

    Special cases:
        - $p=1$: :class:`MAE_Loss`
        - $p=2$: :class:`MSE_Loss`
    """

    p: Final[float]
    r"""The $p$-norm to use."""

    def __init__(
        self,
        p: float = 2.0,
        *,
        channel_weight: Tensor | None = None,
        scaled: bool = False,
        dim: Dim = None,
    ) -> None:
        super().__init__(
            scaled=scaled,
            dim=dim,
            channel_weight=channel_weight,
        )
        self.p = p

    def forward(
        self,
        *,
        predictions: Tensor,  # Float[..., *D], possibly contains NaN
        targets: Tensor,  # Float[..., *D], possibly contains NaN
        mask: Tensor | None = None,  # Bool[..., *D],
        weight: Tensor | None = None,  # Float[...], sample weights
    ) -> Tensor:  # Float[()]
        return lp_loss(
            predictions=predictions,
            targets=targets,
            mask=mask,
            weight=weight,
            p=self.p,
            dim=self.dim,
            channel_weight=self.channel_weight,
            scaled=self.scaled,
        )


class MAE_loss(BaseMetric):
    r"""Mean Absolute Error.

    Given two random vectors $x̂,x∈ℝᴷ$, the mean absolute error is defined as:

    .. math:: 𝖬𝖠𝖤(x̂，x) ≔ 𝔼[‖x̂ - x‖]

    Given $N$ random samples $x_1, …, x_N$ and $x̂_1, …, x̂_N$, it can be estimated as:

    .. math:: 𝖬𝖠𝖤(x̂，x) ∼ \frac{1}{N}∑ₙ₌₁ᴺ ‖x̂ₙ - xₙ‖

    If weights are provided, then the norm $‖z‖² ≔ ∑ₖ wₖ |zₖ|²$ is used.
    """

    def forward(
        self,
        *,
        predictions: Tensor,  # Float[..., *D], possibly contains NaN
        targets: Tensor,  # Float[..., *D], possibly contains NaN
        mask: Tensor | None = None,  # Bool[..., *D],
        weight: Tensor | None = None,  # Float[...], sample weights
    ) -> Tensor:  # Float[()]
        return mae_loss(
            predictions=predictions,
            targets=targets,
            mask=mask,
            weight=weight,
            channel_weight=self.channel_weight,
            scaled=self.scaled,
        )


class MSE_Loss(BaseMetric):
    r"""Mean Square Error.

    Given two random vectors $x̂,x∈ℝᴷ$, the mean square error is defined as:

    .. math:: 𝖬𝖲𝖤(x̂，x) ≔ 𝔼[‖x̂-x‖²] ∼ \frac{1}{N}∑ₙ₌₁ᴺ ‖x̂ₙ - xₙ‖²

    Given $N$ random samples $x_1, …, x_N$ and $x̂_1, …, x̂_N$, it can be estimated as:

    .. math:: 𝖬𝖲𝖤(x̂，x) ∼ \frac{1}{N}∑ₙ₌₁ᴺ ‖x̂ₙ - xₙ‖²

    If the `normalize` option is set to True, then the normalized ℓ²-norm is used instead:

    .. math:: ‖z‖²_{2^*} ≔ \frac{1}{K}∑ₖ₌₁ᴷ zₖ²

    If nan_policy is set to 'omit', then NaN targets are ignored, not counting them as observations.
    In this case, the loss is computed as if the NaN channels would not exist.

    .. math:: ‖z‖²_{2^*} ≔ \frac{1}{∑ₖ mₖ} ∑ₖ₌₁ᴷ [mₖ \? zₖ² : 0]

    Since it could happen that all channels are NaN, the loss is set to zero in this case.

    So, in total, there are 4 variants of the MSE loss:

    Note that this is equivalent to a weighted MSE loss with weights equal to 1.0.

    1. MSE with normalization and NaNs ignored

       .. math:: \frac{1}{N}∑ₙ₌₁ᴺ \frac{1}{∑ₖ mₖ}∑ₖ₌₁ᴷ [mₖ \? |x̂ₙₖ - xₙₖ|² : 0]

    2. MSE with normalization and NaNs counted

       .. math:: \frac{1}{N}∑ₙ₌₁ᴺ \frac{1}{K}∑ₖ₌₁ᴷ |x̂ₙₖ - xₙₖ|²

    3. MSE without normalization and NaNs ignored

       .. math:: \frac{1}{N}∑ₙ₌₁ᴺ ∑ₖ₌₁ᴷ [mₖ \? |x̂ₙₖ - xₙₖ|² : 0]

    4. MSE without normalization and NaNs counted

       .. math:: \frac{1}{N}∑ₙ₌₁ᴺ ∑ₖ₌₁ᴷ |x̂ₙₖ - xₙₖ|²
    """

    def forward(
        self,
        *,
        predictions: Tensor,  # Float[..., *D], possibly contains NaN
        targets: Tensor,  # Float[..., *D], possibly contains NaN
        mask: Tensor | None = None,  # Bool[..., *D],
        weight: Tensor | None = None,  # Float[...], sample weights
    ) -> Tensor:  # Float[()]
        return mse_loss(
            predictions=predictions,
            targets=targets,
            mask=mask,
            weight=weight,
            channel_weight=self.channel_weight,
            scaled=self.scaled,
        )


class RMSE_Loss(BaseMetric):
    r"""Root Mean Square Error.

    Given two random vectors $x̂,x∈ℝᴷ$, the root-mean-square error is defined as:

    .. math:: 𝖱𝖬𝖲𝖤(x̂，x) ≔ \sqrt{𝔼[‖x̂ - x‖²]}

    Given $N$ random samples $x_1, …, x_N$ and $x̂_1, …, x̂_N$, it can be estimated as:

    .. math:: 𝖱𝖬𝖲𝖤(x̂，x) ∼ \sqrt{\frac{1}{N}∑ₙ₌₁ᴺ ‖x̂ₙ - xₙ‖²}
    """

    def forward(
        self,
        *,
        predictions: Tensor,  # Float[..., *D], possibly contains NaN
        targets: Tensor,  # Float[..., *D], possibly contains NaN
        mask: Tensor | None = None,  # Bool[..., *D],
        weight: Tensor | None = None,  # Float[...], sample weights
    ) -> Tensor:  # Float[()]
        return rmse_loss(
            predictions=predictions,
            targets=targets,
            mask=mask,
            weight=weight,
            channel_weight=self.channel_weight,
            scaled=self.scaled,
        )


def q_quantile(
    *,
    predictions: Tensor,  # Float[..., *D]
    targets: Tensor,  # Float[..., *D]
    q: float = 0.5,
    dim: Dim = -1,  # *D
    mask: Tensor | None = None,  # Bool[..., *D],
    weight: Tensor | None = None,  # Float[*D]  # channel weights
) -> Tensor:  # Float[...]
    r"""Return the q-quantile / pinball loss.

    For a scalar valued x, this is just:

    .. math::
        𝖯_q(x̂，x) ≔ \begin{cases}
            \hfill  q⋅|x̂-x| :& x ≥ x̂
            \\  (1-q)⋅|x̂-x| :& x ≤ x̂
        \end{cases}

    For a vector / tensor valued x, we set:

    .. math::  ℓ(x̂，x) ≔ ∑ₖP_q(x̂ₖ，x)
    .. math::  ℓ(x̂，x) ≔ ∑ₖ⟦mₖ \? P_q(x̂ₖ，x) : 0⟧
    .. math::  ℓ(x̂，x) ≔ ∑ₖwₖP_q(x̂ₖ，x)
    .. math::  ℓ(x̂，x) ≔ ∑ₖwₖ⟦mₖ \? P_q(x̂ₖ，x) : 0⟧

    References:
        - | Deep State Space Models for Time Series Forecasting
          | Syama Sundar Rangapuram, Matthias W. Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang,
            Tim Januschowski
          | Advances in Neural Information Processing Systems 31 (NeurIPS 2018)
          | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """
    # simplified formula
    r = targets - predictions
    s = torch.maximum((q - 1) * r, q * r)
    return s


class Q_Quantile(nn.Module):
    r"""The q-quantile / pinball loss.

    .. math::
        𝖯_q(x̂, x) ≔ \begin{cases}
            \hfill  q⋅|x̂-x| :& x ≥ x̂
            \\  (1-q)⋅|x̂-x| :& x ≤ x̂
        \end{cases}

    References:
        - | Deep State Space Models for Time Series Forecasting
          | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """

    q: Final[float]

    def __init__(self, q: float = 0.5):
        super().__init__()
        self.q = q

    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r"""Compute the loss value."""
        return q_quantile(predictions=predictions, targets=targets, q=self.q)
