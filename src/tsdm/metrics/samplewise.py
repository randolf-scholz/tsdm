r"""Implementations of metrics for (non-sequential) data.

These metrics support missing values through the mask argument.
"""

__all__ = [
    "RMSE",
    "MSE",
    "MAE",
    "LP_Loss",
    "Q_Quantile",
    "rmse",
    "mse_loss",
    "mae_loss",
    "lp_loss",
    "q_quantile",
]

from enum import StrEnum
from typing import Any, Final

import torch
from torch import Tensor

from .base import BaseMetric

type Dim = int | tuple[int, ...] | None


class Reduction(StrEnum):
    """Reduction method for metrics."""

    SUM = "sum"
    MEAN = "mean"
    NONE = "none"


UNDEFINED: Final[Any] = object()


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
        For weighted scaled norms, zero-weight entries are treated as masked: the
        denominator is ``sum(weight != 0)`` without a mask and
        ``sum(mask & (weight != 0))`` with one.
    """
    if dim is UNDEFINED:
        # -1 if no weight, else (-weight.ndim, ..., -1)
        dim = (-1,) if weight is None else tuple(range(-weight.ndim, 0))

    dims = tuple(
        range(x.ndim) if dim is None else (dim,) if isinstance(dim, int) else dim
    )

    if weight is not None and weight.ndim != len(dims):
        raise ValueError(f"Expected {weight.ndim=} to equal {len(dims)=}")

    # fmt: off
    match mask, weight:
        case None, None: active = None
        case _,    None: active = mask
        case None, _   : active = weight != 0  # pyrefly: ignore[unsupported-operation]
        case _,    _   : active = mask & (weight != 0)  # pyrefly: ignore[unsupported-operation]
    # fmt: on

    if p == torch.inf:
        raise NotImplementedError
    elif p == -torch.inf:
        raise NotImplementedError
    elif p == 0.0:
        # TODO: if scaled, use geometric mean.
        raise NotImplementedError
    elif p > 0:
        if active is not None:
            x = torch.where(active, x, 0.0)
        r = x.abs().pow(p)
    else:  # p<0
        if active is not None:
            x = torch.where(active, x, 1.0)
        r = x.abs().pow(p)
        if active is not None:
            r = torch.where(active, r, 0.0)

    if weight is not None:
        r = weight * r

    r = torch.sum(r, dim=dims)

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


def lp_loss(
    *,
    predictions: Tensor,
    targets: Tensor,
    p: float = 2.0,
    dim: Dim = -1,
    mask: Tensor | None = None,
    weight: Tensor | None = None,
    scaled: bool = False,
    reduction: Reduction = "mean",
) -> Tensor:
    r"""Compute the $p$-norm.

    .. math:: ℓₚ(x̂，x) ≔ 𝔼[‖x̂ - x‖ₚ]
    """
    norms = lp_norm(
        predictions - targets,
        p=p,
        dim=dim,
        mask=mask,
        weight=weight,
        scaled=scaled,
    )

    match reduction:
        case Reduction.SUM:
            return norms.sum()
        case Reduction.MEAN:
            return norms.mean()
        case Reduction.NONE:
            return norms
        case _:
            raise ValueError(f"Invalid reduction: {reduction}")


def mae_loss(
    *,
    predictions: Tensor,
    targets: Tensor,
    dim: Dim = -1,
    mask: Tensor | None = None,
    weight: Tensor | None = None,
    scaled: bool = False,
) -> Tensor:
    r"""Compute the mean absolute error.

    .. math:: mae(x̂，x) ≔ 𝔼[‖x̂ - x‖₂]
    """
    w = weight
    m = ~targets.isnan()
    r = predictions - targets
    r = torch.where(m, r, 0.0)
    r = r.abs() if w is None else w * r.abs()
    r = torch.sum(r, dim=dim)

    if scaled:
        c = torch.sum(m if w is None else w * m, dim=dim)
    else:
        c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

    r = torch.where(c > 0, r / c, 0.0)

    # aggregate over batch dimensions
    r = torch.mean(r)
    return r


def mse_loss(
    *,
    predictions: Tensor,
    targets: Tensor,
    dim: Dim = -1,
    mask: Tensor | None = None,
    weight: Tensor | None = None,
    scaled: bool = False,
) -> Tensor:
    r"""Compute the mean squared error.

    .. math:: mse(x̂，x) ≔ 𝔼[‖x̂ - x‖₂²]
    """
    w = weight
    m = ~targets.isnan()
    r = predictions - targets
    r = torch.where(m, r, 0.0)
    r = r**2 if w is None else w * r**2
    r = torch.sum(r, dim=dim)  # shape=(..., )

    if scaled:
        c = torch.sum(m if w is None else w * m, dim=dim)
    else:
        c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

    r = torch.where(c > 0, r / c, 0.0)

    # aggregate over batch dimensions
    r = torch.mean(r)
    return r


def rmse(
    *,
    predictions: Tensor,
    targets: Tensor,
    dim: Dim = -1,
    mask: Tensor | None = None,
    weight: Tensor | None = None,
    scaled: bool = False,
) -> Tensor:
    r"""Compute the root mean squared error.

    .. math:: 𝗋𝗆𝗌𝖾(x̂，x) ≔ \sqrt{ 𝔼[‖x̂ - x‖₂²] }
    """
    w = weight
    m = ~targets.isnan()
    r = predictions - targets
    r = torch.where(m, r, 0.0)
    r = r**2 if w is None else w * r**2
    r = torch.sum(r, dim=dim)

    if scaled:
        c = torch.sum(m if w is None else w * m, dim=dim)
    else:
        c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

    r = torch.where(c > 0, r / c, 0.0)

    # aggregate over batch dimensions
    r = torch.mean(r)
    return torch.sqrt(r)


def q_quantile(
    *,
    predictions: Tensor,  # Float[...]
    targets: Tensor,  # Float[...]
    q: float = 0.5,
) -> Tensor:  # Float[...]
    r"""Return the q-quantile.

    For scalar valued x, this is just:

    .. math::
        𝖯_q(x̂, x) ≔ \begin{cases}
            \hfill  q⋅|x̂-x| :& x ≥ x̂
            \\  (1-q)⋅|x̂-x| :& x ≤ x̂
        \end{cases}

    References:
        - | Deep State Space Models for Time Series Forecasting
          | Syama Sundar Rangapuram, Matthias W. Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang,
            Tim Januschowski
          | Advances in Neural Information Processing Systems 31 (NeurIPS 2018)
          | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """
    # simplified formula
    residual = targets - predictions
    return torch.maximum((q - 1) * residual, q * residual)


class MAE(BaseMetric):
    r"""Mean Absolute Error.

    Given two random vectors $x̂,x∈ℝᴷ$, the mean absolute error is defined as:

    .. math:: 𝖬𝖠𝖤(x̂，x) ≔ 𝔼[‖x̂ - x‖]

    Given $N$ random samples $x_1, …, x_N$ and $x̂_1, …, x̂_N$, it can be estimated as:

    .. math:: 𝖬𝖠𝖤(x̂，x) ∼ \frac{1}{N}∑ₙ₌₁ᴺ ‖x̂ₙ - xₙ‖

    If weights are provided, then the norm $‖z‖² ≔ ∑ₖ wₖ |zₖ|²$ is used.
    """

    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r""".. signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        return mae_loss(
            predictions=predictions,
            targets=targets,
            weight=self.weight,
            scaled=self.normalize,
        )


class MSE(BaseMetric):
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

    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r""".. signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        return mse_loss(
            predictions=predictions,
            targets=targets,
            weight=self.weight,
            scaled=self.normalize,
        )


class RMSE(BaseMetric):
    r"""Root Mean Square Error.

    Given two random vectors $x̂,x∈ℝᴷ$, the root-mean-square error is defined as:

    .. math:: 𝖱𝖬𝖲𝖤(x̂，x) ≔ \sqrt{𝔼[‖x̂ - x‖²]}

    Given $N$ random samples $x_1, …, x_N$ and $x̂_1, …, x̂_N$, it can be estimated as:

    .. math:: 𝖱𝖬𝖲𝖤(x̂，x) ∼ \sqrt{\frac{1}{N}∑ₙ₌₁ᴺ ‖x̂ₙ - xₙ‖²}
    """

    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r""".. signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        return rmse(
            predictions=predictions,
            targets=targets,
            weight=self.weight,
            scaled=self.normalize,
        )


class LP_Loss(BaseMetric):
    r"""$Lᵖ$ Loss.

    Given two random vectors $x̂,x∈ℝᴷ$, the $Lᵖ$-loss is defined as:

    .. math:: 𝖱𝖬𝖲𝖤(x̂，x) ≔ \sqrt[p]{𝔼[‖x̂ - x‖ᵖ]}

    Given $N$ random samples $x_1, …, x_N$ and $x̂_1, …, x̂_N$, it can be estimated as:

    .. math:: 𝖱𝖬𝖲𝖤(x̂，x) ∼ \sqrt[p]{\frac{1}{N}∑ₙ₌₁ᴺ ‖x̂ₙ - xₙ‖ᵖ}

    Special cases:
        - $p=1$: :class:`MAE`
        - $p=2$: :class:`RMSE`
    """

    p: Final[float]
    r"""The $p$-norm to use."""

    def __init__(
        self,
        p: float = 2.0,
        *,
        weight: Tensor | None = None,
        normalize: bool = False,
        dim: Dim = None,
        learnable: bool = False,
    ) -> None:
        super().__init__(
            normalize=normalize,
            dim=dim,
            weight=weight,
            learnable=learnable,
        )
        self.p = p

    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r""".. signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        return lp_loss(
            predictions=predictions,
            targets=targets,
            p=self.p,
            weight=self.weight,
            normalize=self.normalize,
        )


class Q_Quantile(BaseMetric):
    r"""The q-quantile.

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

    def __init__(self, q: float = 0.5, *, dim: Dim = None):
        super().__init__(dim=dim)
        self.q = q

    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r"""Compute the loss value."""
        return q_quantile(predictions=predictions, targets=targets, q=self.q)
