r"""Loss functions that by definition include aggregation over samples.

This includes for instance the standard RMSE loss.
These should not be evaluated on mini-batches, instead they should only be estimated
on the full dataset.
"""

__all__ = ["quantile_loss"]

from typing import Final

from torch import Tensor, nn

from .base import BaseLoss
from .samplewise import Dim, Reduction, lp_loss, quantile_error


def quantile_loss(
    *,
    predictions: Tensor,
    targets: Tensor,
    q: float = 0.5,
    dim: int = -1,
    relative: bool = True,
) -> Tensor:
    r"""Compute the QL loss, based on relative q-quantile values.

    .. math:: ℓ(x̂，x) ≔ ∑  (2 ∑ₖ⟦mₖ \? P_q(x̂ₖ-xₖ) : 0⟧ / ∑ₖ⟦mₖ \? |xₖ| : 0⟧)

    Args:
        predictions: The predicted values.
        targets: The target values.
        q: The quantile level.
        dim: The dimension along which to compute the loss.
        relative: Whether to compute the relative loss.

    References:
        - | Deep State Space Models for Time Series Forecasting
          | Syama Sundar Rangapuram et al.
          | Advances in Neural Information Processing Systems 31 (NeurIPS 2018)
          | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """
    return quantile_error(predictions - targets, q=q, dim=dim)


class Quantile_Loss(nn.Module):
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
        return quantile_error(predictions - targets, q=self.q)


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
    return Reduction(reduction).apply(squared_norms, weight=weight).sqrt()


class RMSE_Loss(BaseLoss):
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
