r"""Implementation of losses that are applies pointwise.

These can still be used for sequential data, however, note that in this case
the sequence dimension is simply treated as any other a batch dimension.
"""

__all__ = [
    "SeparableLoss",
    "L2_Loss",
    "LP_Loss",
    "MAE_Loss",
    "MSE_Loss",
    "RMSE_Loss",
    # losses
    "l2_loss",
    "lp_loss",
    "mae_loss",
    "mse_loss",
    "rmse_loss",
    # error metrics
    "lp_error",
]

from typing import Final

from torch import Tensor, nn

from .base import BaseLoss, Loss, Penalty, Reduction
from .penalties.pointwise import lp_norm


def lp_error(
    *,
    predictions: Tensor,  # Float[..., *D]
    targets: Tensor,  # Float[..., *D],
    dim: int | tuple[int, ...],  # *D
    p: float = 2.0,
    mask: Tensor | None = None,  # Bool[..., *D]
    feature_weights: Tensor | None = None,  # Float[..., *D] or Float[*D], non-negative
    scaled: bool = False,
    relative: bool = False,
) -> Tensor:  # Float[...]
    norms = lp_norm(
        predictions - targets,
        p=p,
        dim=dim,
        mask=mask,
        weight=feature_weights,
        scaled=scaled,
    )
    if relative:
        norms = norms / lp_norm(
            targets,
            p=p,
            dim=dim,
            mask=mask,
            weight=feature_weights,
            scaled=scaled,
        )
    return norms


def lp_loss(
    *,
    predictions: Tensor,  # Float[..., *D]
    targets: Tensor,  # Float[..., *D],
    p: float = 2.0,
    dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., *D],
    sample_weights: Tensor | None = None,  # Float[...], sample weights
    feature_weights: Tensor | None = None,  # Float[..., *D], channel weights
    reduction: Reduction = Reduction.MEAN,
    scaled: bool = False,
    relative: bool = False,
) -> Tensor:  # Float[()]
    r"""Compute the sample-weighted $p$-norm loss.

    .. math:: ℓₚ(x̂，x) ≔ \E[‖x̂ - x‖ₚ]

    ``channel_weight`` is applied within the $p$-norm along ``dim``. ``weight``
    is applied to the resulting per-sample losses before ``reduction``.

    Args:
        predictions: Predicted values.
        targets: Target values.
        p: Order of the norm.
        dim: Tensor axes reduced within each sample.
        mask: Boolean mask indicating valid values.
        sample_weights: Optional weights applied to the per-sample losses.
        feature_weights: Optional weights applied within each sample norm.
        reduction: Aggregation applied to the per-sample losses.
        scaled: Whether to scale each norm by its number of active entries.
        relative: If ``True``, divide each residual norm by the corresponding
            target norm, computing $‖x̂ - x‖ₚ / ‖x‖ₚ$.
    """
    errors = lp_error(
        predictions=predictions,
        targets=targets,
        p=p,
        dim=dim,
        mask=mask,
        feature_weights=feature_weights,
        scaled=scaled,
        relative=relative,
    )
    return Reduction(reduction).apply(errors, weights=sample_weights)


def mae_loss(
    *,
    predictions: Tensor,  # Float[..., *D]
    targets: Tensor,  # Float[..., *D]
    dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., *D],
    sample_weights: Tensor | None = None,  # Float[...], sample weights
    feature_weights: Tensor | None = None,  # Float[..., *D], channel weights
    reduction: Reduction = Reduction.MEAN,
    scaled: bool = False,
) -> Tensor:  # Float[()]
    r"""Compute the sample-weighted mean absolute error.

    .. math:: \mae(x̂，x) ≔ \E[‖x̂ - x‖₁]

    This is minimized by the marginal median of the distribution.
    """
    errors = lp_error(
        predictions=predictions,
        targets=targets,
        p=1.0,
        dim=dim,
        mask=mask,
        feature_weights=feature_weights,
        scaled=scaled,
    )
    return Reduction(reduction).apply(errors, weights=sample_weights)


def l2_loss(
    *,
    predictions: Tensor,  # Float[..., *D]
    targets: Tensor,  # Float[..., *D]
    dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., *D],
    sample_weights: Tensor | None = None,  # Float[...], sample weights
    feature_weights: Tensor | None = None,  # Float[..., *D], channel weights
    scaled: bool = False,
    reduction: Reduction = Reduction.MEAN,
) -> Tensor:  # Float[()]
    r"""Compute the mean euclidean distance.

    .. math:: ℓ₂(x̂，x) ≔ \E[‖x̂ - x‖₂]

    This is minimized by the geometric median of the distribution.
    """
    errors = lp_error(
        predictions=predictions,
        targets=targets,
        p=2.0,
        dim=dim,
        mask=mask,
        feature_weights=feature_weights,
        scaled=scaled,
    )
    return Reduction(reduction).apply(errors, weights=sample_weights)


def mse_loss(
    *,
    predictions: Tensor,  # Float[..., *D]
    targets: Tensor,  # Float[..., *D]
    dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., *D],
    sample_weights: Tensor | None = None,  # Float[...], sample weights
    feature_weights: Tensor | None = None,  # Float[..., *D], channel weights
    scaled: bool = False,
    reduction: Reduction = Reduction.MEAN,
) -> Tensor:  # Float[()]
    r"""Compute the sample-weighted mean squared error.

    .. math:: \mse(x̂，x) ≔ \E[‖x̂ - x‖₂²]

    This is minimized by the mean of the distribution.
    """
    errors = lp_error(
        predictions=predictions,
        targets=targets,
        p=2.0,
        dim=dim,
        mask=mask,
        feature_weights=feature_weights,
        scaled=scaled,
    ).square()
    return Reduction(reduction).apply(errors, weights=sample_weights)


def rmse_loss(
    *,
    predictions: Tensor,  # Float[..., *D]
    targets: Tensor,  # Float[..., *D]
    dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., *D],
    sample_weights: Tensor | None = None,  # Float[...], sample weights
    feature_weights: Tensor | None = None,  # Float[..., *D], channel weights
    reduction: Reduction = Reduction.MEAN,
    scaled: bool = False,
) -> Tensor:  # Float[()]
    r"""Compute the sample-weighted root mean squared error.

    .. math:: \𝗋𝗆𝗌𝖾(x̂，x) ≔ \sqrt{ \E[‖x̂ - x‖₂²] }
    """
    errors = lp_error(
        predictions=predictions,
        targets=targets,
        p=2.0,
        dim=dim,
        mask=mask,
        feature_weights=feature_weights,
        scaled=scaled,
    ).square()
    return Reduction(reduction).apply(errors, weights=sample_weights).sqrt()


class SeparableLoss(nn.Module, Loss):
    r"""Base class for separable loss function with reduction over samples.

    Note:
        A loss is separable if there is no interaction between batch elements,
        i.e. $ℒ(x̂, x) = Reduce(wₙℓ(x̂ₙ, xₙ)ₙ})$ for some per sample loss ℓ
        and a reduction function Reduce.
    """

    penalty: Penalty
    r"""CONST: The penalty function to apply."""
    reduction: Reduction
    r"""CONST: The reduction to apply over the samples."""

    def __init__(
        self,
        *,
        feature_dim: int | tuple[int, ...],
    ) -> None:
        super().__init__()
        self.feature_dim = (
            (feature_dim,) if isinstance(feature_dim, int) else feature_dim
        )

    def forward(
        self,
        *,
        predictions: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
        sample_weights: Tensor | None = None,
    ) -> Tensor:
        errors = self.penalty(
            predictions=predictions,
            targets=targets,
            mask=mask,
            feature_dim=self.feature_dim,
        )
        return self.reduction.apply(errors, weights=sample_weights)


class LP_Loss(BaseLoss):
    r"""$Lᵖ$ Loss."""

    p: Final[float]
    r"""The $p$-norm to use."""

    def __init__(
        self,
        p: float = 2.0,
        *,
        feature_weights: Tensor | None = None,
        scaled: bool = False,
        dim: int | tuple[int, ...] = -1,
    ) -> None:
        super().__init__(
            scaled=scaled,
            dim=dim,
            feature_weights=feature_weights,
        )
        self.p = p

    def forward(
        self,
        *,
        predictions: Tensor,  # Float[..., *D], possibly contains NaN
        targets: Tensor,  # Float[..., *D], possibly contains NaN
        mask: Tensor | None = None,  # Bool[..., *D],
        sample_weights: Tensor | None = None,  # Float[...], sample weights
    ) -> Tensor:  # Float[()]
        return lp_loss(
            predictions=predictions,
            targets=targets,
            mask=mask,
            sample_weights=sample_weights,
            p=self.p,
            dim=self.dim,
            feature_weights=self.feature_weights,
            scaled=self.scaled,
        )


class MAE_Loss(BaseLoss):
    r"""Mean Absolute Error.

    Given two random vectors $x̂,x∈ℝᴷ$, the mean absolute error is defined as:

    .. math:: \mae(x̂，x) ≔ \E[‖x̂ - x‖₁]

    Given $N$ random samples $x_1, …, x_N$ and $x̂_1, …, x̂_N$, it can be estimated as:

    .. math:: \mae(x̂，x) ∼ \frac{1}{N}∑ₙ₌₁ᴺ ‖x̂ₙ - xₙ‖

    If weights are provided, then the norm $‖z‖² ≔ ∑ₖ wₖ\abs{zₖ}²$ is used.
    """

    def forward(
        self,
        *,
        predictions: Tensor,  # Float[..., *D], possibly contains NaN
        targets: Tensor,  # Float[..., *D], possibly contains NaN
        mask: Tensor | None = None,  # Bool[..., *D],
        sample_weights: Tensor | None = None,  # Float[...], sample weights
    ) -> Tensor:  # Float[()]
        return mae_loss(
            predictions=predictions,
            targets=targets,
            mask=mask,
            sample_weights=sample_weights,
            feature_weights=self.feature_weights,
            scaled=self.scaled,
        )


class L2_Loss(BaseLoss):
    r"""Mean Euclidean Distance.

    Given two random vectors $x̂,x∈ℝᴷ$, the median loss:

    .. math:: ℓ₂(x̂，x) ≔ \E[‖x̂ - x‖₂]

    This is minimized by the geometric median of the distribution.
    """

    def forward(
        self,
        *,
        predictions: Tensor,  # Float[..., *D], possibly contains NaN
        targets: Tensor,  # Float[..., *D], possibly contains NaN
        mask: Tensor | None = None,  # Bool[..., *D],
        sample_weights: Tensor | None = None,  # Float[...], sample weights
    ) -> Tensor:  # Float[()]
        return l2_loss(
            predictions=predictions,
            targets=targets,
            mask=mask,
            sample_weights=sample_weights,
            feature_weights=self.feature_weights,
            scaled=self.scaled,
        )


class MSE_Loss(BaseLoss):
    r"""Mean Squared Error.

    Given two random vectors $x̂,x∈ℝᴷ$, the MSE loss:

    .. math:: \mse(x̂，x) ≔ \E[‖x̂ - x‖₂²]

    This is minimized by the mean of the distribution.
    """

    def forward(
        self,
        *,
        predictions: Tensor,  # Float[..., *D], possibly contains NaN
        targets: Tensor,  # Float[..., *D], possibly contains NaN
        mask: Tensor | None = None,  # Bool[..., *D],
        sample_weights: Tensor | None = None,  # Float[...], sample weights
    ) -> Tensor:  # Float[()]
        return mse_loss(
            predictions=predictions,
            targets=targets,
            mask=mask,
            sample_weights=sample_weights,
            feature_weights=self.feature_weights,
            scaled=self.scaled,
        )


class RMSE_Loss(BaseLoss):
    r"""Root Mean Square Error.

    Given two random vectors $x̂,x∈ℝᴷ$, the root-mean-square error is defined as:

    .. math:: \rmse(x̂，x) ≔ \sqrt{\E[‖x̂ - x‖²]}

    Given $N$ random samples $x_1, …, x_N$ and $x̂_1, …, x̂_N$, it can be estimated as:

    .. math:: \rmse(x̂，x) ∼ \sqrt{\frac{1}{N}∑ₙ₌₁ᴺ ‖x̂ₙ - xₙ‖²}
    """

    def forward(
        self,
        *,
        predictions: Tensor,  # Float[..., *D], possibly contains NaN
        targets: Tensor,  # Float[..., *D], possibly contains NaN
        mask: Tensor | None = None,  # Bool[..., *D],
        sample_weights: Tensor | None = None,  # Float[...], sample weights
    ) -> Tensor:  # Float[()]
        return rmse_loss(
            predictions=predictions,
            targets=targets,
            mask=mask,
            sample_weights=sample_weights,
            feature_weights=self.feature_weights,
            scaled=self.scaled,
        )
