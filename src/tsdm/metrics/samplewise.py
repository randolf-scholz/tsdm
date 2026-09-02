r"""Implementations of metrics for (non-sequential) data."""

__all__ = [
    "RMSE",
    "MSE",
    "MAE",
    "LP",
    "Q_Quantile",
    "rmse",
    "mse",
    "mae",
    "lp",
    "q_quantile",
]

from typing import Final

import torch
from torch import Tensor

from tsdm.types.aliases import Axis

from .base import BaseMetric


def mae(
    *,
    predictions: Tensor,
    targets: Tensor,
    dim: Axis = -1,
    weight: Tensor | None = None,
    normalize: bool = False,
) -> Tensor:
    w = weight
    m = ~targets.isnan()
    r = predictions - targets
    r = torch.where(m, r, 0.0)
    r = r.abs() if w is None else w * r.abs()
    r = torch.sum(r, dim=dim)

    if normalize:
        c = torch.sum(m if w is None else w * m, dim=dim)
    else:
        c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

    r = torch.where(c > 0, r / c, 0.0)

    # aggregate over batch dimensions
    r = torch.mean(r)
    return r


def mse(
    *,
    predictions: Tensor,
    targets: Tensor,
    dim: Axis = -1,
    weight: Tensor | None = None,
    normalize: bool = False,
) -> Tensor:
    r"""Compute the MSE."""
    w = weight
    m = ~targets.isnan()
    r = predictions - targets
    r = torch.where(m, r, 0.0)
    r = r**2 if w is None else w * r**2
    r = torch.sum(r, dim=dim)  # shape=(..., )

    if normalize:
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
    dim: Axis = -1,
    weight: Tensor | None = None,
    normalize: bool = False,
) -> Tensor:
    r"""Compute the RMSE.

    .. math:: 𝗋𝗆𝗌𝖾(x̂，x) ≔ \sqrt{𝔼[‖x̂ - x‖²]}
    """
    w = weight
    m = ~targets.isnan()
    r = predictions - targets
    r = torch.where(m, r, 0.0)
    r = r**2 if w is None else w * r**2
    r = torch.sum(r, dim=dim)

    if normalize:
        c = torch.sum(m if w is None else w * m, dim=dim)
    else:
        c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

    r = torch.where(c > 0, r / c, 0.0)

    # aggregate over batch dimensions
    r = torch.mean(r)
    return torch.sqrt(r)


def lp(
    *,
    predictions: Tensor,
    targets: Tensor,
    p: float = 2.0,
    dim: Axis = -1,
    weight: Tensor | None = None,
    normalize: bool = False,
) -> Tensor:
    r"""Compute the $p$-norm."""
    w = weight
    m = ~targets.isnan()
    r = predictions - targets
    r = torch.where(m, r, 0.0)
    r = r**p if w is None else w * r**p
    r = torch.sum(r, dim=dim)

    if normalize:
        c = torch.sum(m if w is None else w * m, dim=dim)
    else:
        c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

    r = torch.where(c > 0, r / c, 0.0)

    # aggregate over batch dimensions
    r = torch.mean(r)
    return torch.pow(r, 1 / p)


def q_quantile(
    *,
    predictions: Tensor,
    targets: Tensor,
    q: float = 0.5,
) -> Tensor:
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

    @torch.compile(fullgraph=True)
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r""".. signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        return mae(
            predictions=predictions,
            targets=targets,
            weight=self.weight,
            normalize=self.normalize,
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

    @torch.compile(fullgraph=True)
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r""".. signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        return mse(
            predictions=predictions,
            targets=targets,
            weight=self.weight,
            normalize=self.normalize,
        )


class RMSE(BaseMetric):
    r"""Root Mean Square Error.

    Given two random vectors $x̂,x∈ℝᴷ$, the root-mean-square error is defined as:

    .. math:: 𝖱𝖬𝖲𝖤(x̂，x) ≔ \sqrt{𝔼[‖x̂ - x‖²]}

    Given $N$ random samples $x_1, …, x_N$ and $x̂_1, …, x̂_N$, it can be estimated as:

    .. math:: 𝖱𝖬𝖲𝖤(x̂，x) ∼ \sqrt{\frac{1}{N}∑ₙ₌₁ᴺ ‖x̂ₙ - xₙ‖²}
    """

    @torch.compile(fullgraph=True)
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r""".. signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        return rmse(
            predictions=predictions,
            targets=targets,
            weight=self.weight,
            normalize=self.normalize,
        )


class LP(BaseMetric):
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
        axis: Axis = None,
        learnable: bool = False,
    ) -> None:
        super().__init__(
            normalize=normalize,
            axis=axis,
            weight=weight,
            learnable=learnable,
        )
        self.p = p

    @torch.compile(fullgraph=True)
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r""".. signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        return lp(
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

    @torch.compile(fullgraph=True)
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r"""Compute the loss value."""
        return q_quantile(predictions=predictions, targets=targets)
