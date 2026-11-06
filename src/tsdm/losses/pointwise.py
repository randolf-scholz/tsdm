r"""Losses applied independently to vector-valued samples.

The selected feature axes are reduced within each sample, followed by a batch
reduction. For sequence data, the time axis is treated as a batch axis unless
it is included explicitly in the feature axes.
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
    p: float,
    dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., *D],
    sample_weights: Tensor | None = None,  # Float[...], sample weights
    feature_weights: Tensor | None = None,  # Float[..., *D], channel weights
    reduction: Reduction = Reduction.MEAN,
    scaled: bool = False,
    relative: bool = False,
) -> Tensor:  # Float[()]
    r"""Compute the sample-weighted $p$-norm loss.

    For unweighted samples with the default mean reduction and unscaled norms:

    $$ ℓₚ(x̂,x) ≔ 𝔼[‖x̂ - x‖ₚ] $$

    `feature_weights` applies within the norm along `dim`. `sample_weights` applies
    to the per-sample losses through `Reduction`. With `relative=True`, divide each
    residual norm by the corresponding target norm; zero target norms can produce
    NaN or infinity. Masking is explicit; NaNs are not detected automatically.

    Args:
        predictions: Predicted values.
        targets: Target values.
        p: Finite nonzero norm order.
        dim: Tensor axes reduced within each sample.
        mask: Boolean mask selecting valid entries.
        sample_weights: Optional weights for the outer sample reduction.
        feature_weights: Optional nonnegative weights within each sample norm.
        reduction: Aggregation applied to per-sample losses.
        scaled: Whether to divide the powered sum by the number of active entries.
        relative: Whether to divide by the corresponding target norm ($‖x̂ - x‖ₚ / ‖x‖ₚ$).
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

    $$ \mae(x̂，x) ≔ 𝔼[‖x̂ - x‖₁] $$

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
    r"""Compute the mean Euclidean distance.

    $$ ℓ₂(x̂，x) ≔ 𝔼[‖x̂ - x‖₂] $$

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

    $$ \mse(x̂，x) ≔ 𝔼[‖x̂ - x‖₂²] $$

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

    $$ \rmse(x̂，x) ≔ \sqrt{ 𝔼[‖x̂ - x‖₂²] } $$
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
    r"""Base class for separable losses with a reduction over samples.

    Apply a penalty independently to each prediction-target pair, then combine the
    resulting values using `Reduction`, optionally with sample weights:

    $$ 𝓛(x̂,x) = \op{Reduce}\bigl((ℓ(x̂ₙ,xₙ))ₙ\bigr) $$

    The penalty operates within a sample; the outer reduction combines samples.
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
            dim=self.feature_dim,
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
    r"""Mean absolute error between vector-valued samples.

    For the default unscaled norm and an unweighted sample mean:

    $$ \mae(x̂,x) ≔ 𝔼[‖x̂ - x‖₁] $$

    The empirical estimate from $N$ samples is

    $$ \widehat{\mae} = N⁻¹∑ₙ₌₁ᴺ ‖x̂ₙ - xₙ‖₁ $$

    Feature weights give the inner sum $∑ₖ wₖ|zₖ|$. With `scaled=True`, divide
    that sum by the number of active features. Sample weights affect the outer mean.
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
    r"""Mean Euclidean distance between vector-valued samples.

    For the default unscaled norm and an unweighted sample mean:

    $$ ℓ₂(x̂,x) ≔ 𝔼[‖x̂ - x‖₂] $$

    As a function of a constant prediction, this objective is minimized by a
    geometric median of the target distribution.
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
    r"""Mean squared error between vector-valued samples.

    For the default unscaled norm and an unweighted sample mean:

    $$ \mse(x̂,x) ≔ 𝔼[‖x̂ - x‖₂²] $$

    As a function of a constant prediction, this objective is minimized by the
    mean of the target distribution.
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
    r"""Root mean squared error between vector-valued samples.

    For the default unscaled norm and an unweighted sample mean:

    $$ \rmse(x̂,x) ≔ \sqrt{𝔼[‖x̂ - x‖₂²]} $$

    The empirical estimate from $N$ samples is

    $$ \widehat{\rmse} = \sqrt{N⁻¹∑ₙ₌₁ᴺ ‖x̂ₙ - xₙ‖₂²} $$

    The square root is applied after the outer sample reduction.
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
