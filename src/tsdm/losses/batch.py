r"""Loss functions that by definition include aggregation over samples.

This includes for instance the standard RMSE loss.
These should not be evaluated on mini-batches, instead they should only be estimated
on the full dataset.
"""

__all__ = [
    "RMSE_Loss",
    "rmse_loss",
]


from torch import Tensor

from .base import BaseLoss, Dim, Reduction
from .samplewise import lp_loss


def rmse_loss(
    *,
    predictions: Tensor,  # Float[..., *D]
    targets: Tensor,  # Float[..., *D]
    dim: Dim = -1,
    mask: Tensor | None = None,  # Bool[..., *D],
    sample_weight: Tensor | None = None,  # Float[...], sample weights
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
    return Reduction(reduction).apply(squared_norms, weight=sample_weight).sqrt()


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
            sample_weight=weight,
            channel_weight=self.channel_weight,
            scaled=self.scaled,
        )
