r"""Loss functions for sequential data.

We have some data sets where the number of samples can vary drastically between channels.
In that case, since we want the model still to be able to predict different channels,
one should normalize the per-channel loss.

a general Lₚ loss could look like:

Nₙₖ = ∑ₜmₙₜₖ
Iₙₖ = [Nₙₖ > 0]
Sₖ = ∑ₙIₙₖ

ℓₚ = ( ∑ₙ (1/K) ∑ₖ ( [Nₙₖ>0]  / )    ∑ₜ mₙₜₖ|xₙₜₖ|ᵖ )^{1/p}

Iₙₖ = Anyₜ(mₙₜₖ) = 1-∏ₜ(1-mₙₜₖ)  # was any value observed in channel k for batch element n
Kₙ = ∑ₖ Iₙₖ  # number of channels with at least one observation for batch element n.
Sₖ = ∑ₙ Iₙₖ  # number of batch elements with at least one obervation in channel k.

Wₙ = ∑ₖIₙₖwₖ  # per sample total channel weight.




Since batch elements can be of different lengths, we need to be careful about the
computation of the loss. We add a mask argument.

Typically, the mask should agree with the arguments provided by `targets` and `predictions`
should provide arguments for these positions.
"""

__all__ = [
    # Classes
    "SeparableLoss",
    "L2_Loss",
    "LP_Loss",
    "MAE_Loss",
    "MSE_Loss",
    "ND_Loss",
    "NRMSE_Loss",
    "RMSE_Loss",
    # losses
    "l2_loss",
    "lp_loss",
    "mae_loss",
    "mse_loss",
    "nd_loss",
    "nrmse_loss",
    "rmse_loss",
]

from torch import Tensor

from .base import BaseSequenceLoss, Reduction, SequencePenalty
from .penalties.sequence import (
    Normalization,
    l2_error,
    lp_error,
    mae,
    mse,
    nd,
    nrmse,
    rmse,
)


def lp_loss(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    time_dim: int,
    channel_dim: int | tuple[int, ...],  # *D
    # optional args
    p: float = 2.0,
    relative: bool = False,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
    reduction: Reduction = Reduction.MEAN,
    normalization: Normalization | Tensor | None = Normalization.CHANNEL_PREVALENCE,
    # weights:
    time_weights: Tensor | None = None,  # Float[..., $N]
    channel_weights: Tensor | None = None,  # Float[*D]
    sample_weights: Tensor | None = None,  # Float[...]
) -> Tensor:  # Float[()]
    r"""Compute the reduced time-normalized $p$-norm of prediction residuals.

    .. math:: ℓₚ(x̂, x) ≔ 𝔼[‖x̂ - x‖ₚ]

    The arguments other than ``predictions``, ``targets``, and ``reduction``
    are forwarded to :func:`lp_norm`. ``reduction`` aggregates over the batch
    dimensions of the resulting norms.

    Args:
        predictions: Predicted time series.
        targets: Target time series.
        time_dim: Tensor axis of the time dimension.
        channel_dim: Tensor axes of the channel dimensions.

        p: Order of the norm.
        relative: If ``True``, divide each residual norm by the corresponding
            target norm, computing $‖x̂ - x‖ₚ / ‖x‖ₚ$.
        mask: Optional boolean mask indicating valid values.

        reduction: Aggregation applied to the per-sequence losses.
        normalization: Time-series normalization applied within each norm.

        time_weights: Optional weights applied to time steps.
        channel_weights: Optional weights applied within each channel norm.
        sample_weights: Optional weights applied to each sample.
    """
    errors = lp_error(
        predictions=predictions,
        targets=targets,
        p=p,
        mask=mask,
        channel_dim=channel_dim,
        channel_weights=channel_weights,
        time_dim=time_dim,
        time_weights=time_weights,
        normalization=normalization,
        relative=relative,
    )
    return Reduction(reduction).apply(errors, weights=sample_weights)


def l2_loss(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    # optional args
    time_dim: int = -2,
    channel_dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
    reduction: Reduction = Reduction.MEAN,
    sample_weights: Tensor | None = None,  # Float[...]
) -> Tensor:  # Float[...]
    errors = l2_error(
        predictions=predictions,
        targets=targets,
        time_dim=time_dim,
        channel_dim=channel_dim,
        mask=mask,
    )
    return Reduction(reduction).apply(errors, weights=sample_weights)


def mae_loss(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    # optional args
    time_dim: int = -2,
    channel_dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
    reduction: Reduction = Reduction.MEAN,
    sample_weights: Tensor | None = None,  # Float[...]
) -> Tensor:  # Float[...]
    errors = mae(
        predictions=predictions,
        targets=targets,
        time_dim=time_dim,
        channel_dim=channel_dim,
        mask=mask,
    )
    return Reduction(reduction).apply(errors, weights=sample_weights)


def mse_loss(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    # optional args
    time_dim: int = -2,
    channel_dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
    reduction: Reduction = Reduction.MEAN,
    sample_weights: Tensor | None = None,  # Float[...]
) -> Tensor:  # Float[...]
    errors = mse(
        predictions=predictions,
        targets=targets,
        time_dim=time_dim,
        channel_dim=channel_dim,
        mask=mask,
    )
    return Reduction(reduction).apply(errors, weights=sample_weights)


def rmse_loss(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    # optional args
    time_dim: int = -2,
    channel_dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
    reduction: Reduction = Reduction.MEAN,
    sample_weights: Tensor | None = None,  # Float[...]
) -> Tensor:  # Float[...]
    errors = rmse(
        predictions=predictions,
        targets=targets,
        time_dim=time_dim,
        channel_dim=channel_dim,
        mask=mask,
    )
    return Reduction(reduction).apply(errors, weights=sample_weights)


def nrmse_loss(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    # optional args
    time_dim: int = -2,
    channel_dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
    reduction: Reduction = Reduction.MEAN,
    sample_weights: Tensor | None = None,  # Float[...]
) -> Tensor:  # Float[...]
    errors = nrmse(
        predictions=predictions,
        targets=targets,
        time_dim=time_dim,
        channel_dim=channel_dim,
        mask=mask,
    )
    return Reduction(reduction).apply(errors, weights=sample_weights)


def nd_loss(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    # optional args
    time_dim: int = -2,
    channel_dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
    reduction: Reduction = Reduction.MEAN,
    sample_weights: Tensor | None = None,  # Float[...]
) -> Tensor:  # Float[...]
    errors = nd(
        predictions=predictions,
        targets=targets,
        time_dim=time_dim,
        channel_dim=channel_dim,
        mask=mask,
    )
    return Reduction(reduction).apply(errors, weights=sample_weights)


class SeparableLoss(BaseSequenceLoss):
    r"""Base class for separable time-series loss with a reduction over samples.

    Note:
        A loss is separable if there is no interaction between batch elements,
        i.e. $ℒ(x̂, x) = Reduce(ℓ(x̂ₙ, xₙ)ₙ})$ for some per sample loss ℓ
        and a reduction function Reduce.
    """

    penalty: SequencePenalty
    r"""CONST: The penalty function to apply per sequence pair."""
    reduction: Reduction
    r"""CONST: The reduction to apply over the samples."""

    def __init__(
        self,
        *,
        time_dim: int,
        channel_dim: int | tuple[int, ...],
        penalty: SequencePenalty | None = None,
        reduction: Reduction | None = None,
    ) -> None:
        super().__init__(time_dim=time_dim, channel_dim=channel_dim)

        match penalty, getattr(self, "penalty", None):
            case None, None:
                raise TypeError("Expected a `penalty` argument.")
            case None, _:  # use class attribute
                assert callable(self.penalty)
            case _, None:  # use instance attribute
                assert callable(penalty)
                self.penalty = penalty
            case _, _:
                raise TypeError(
                    "Cannot override class attribute `penalty` with an instance attribute."
                )

        match reduction, getattr(self, "reduction", None):
            case None, None:
                raise TypeError("Expected a `reduction` argument.")
            case None, _:  # use class attribute
                assert isinstance(self.reduction, Reduction)
            case _, None:  # use instance attribute
                assert isinstance(reduction, Reduction)
                self.reduction = reduction
            case _, _:
                raise TypeError(
                    "Cannot override class attribute `reduction` with an instance attribute."
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
            time_dim=self.time_dim,
            channel_dim=self.channel_dim,
        )
        return self.reduction.apply(errors, weights=sample_weights)


class L2_Loss(SeparableLoss):
    r"""Mean Euclidean error.

    .. math:: ℓ(x̂，x) ≔ \tfrac{1}{T} ∑ₜ ‖x̂ₜ - x₂‖₂ = ‖x̂ - x‖_{2,1⁎}

    the masked version is defined as

    .. math:: ℓ(x̂，x) ≔ \tfrac{1}{∑ₛ⋁ₖmₛₖ} ∑ₜ‖mₜ⊙(x̂ₜ - xₜ)‖₂

    where $m ⊙ z ≔ \where(m, z, 0)$ denotes elementwise masked selection. Thus,
    masked ``NaN`` and infinite entries do not contribute; in particular, $0 ⊙ 𝙽𝚊𝙽 = 0$.
    """

    penalty = staticmethod(l2_error)
    reduction = Reduction.MEAN


class MAE_Loss(SeparableLoss):
    r"""Mean absolute error.

    .. math:: \mae(x̂，x) ≔ \tfrac{1}{T} ∑ₜ ‖x̂ₜ - x₂‖₁

    the masked version is defined as

    .. math:: \mae(x̂，x) ≔ \tfrac{1}{∑ₜ⋁ₖmₜₖ} ∑ₜ ‖mₜ⊙(x̂ₜ - xₜ)‖₁

    where $m ⊙ z ≔ \where(m, z, 0)$ denotes elementwise masked selection. Thus,
    masked ``NaN`` and infinite entries do not contribute; in particular, $0 ⊙ 𝙽𝚊𝙽 = 0$.
    """

    penalty = staticmethod(mae)
    reduction = Reduction.MEAN


class MSE_Loss(SeparableLoss):
    r"""Mean square error.

    .. math:: \𝗆𝗌𝖾(x̂，x) ≔ \tfrac{1}{T} ∑ₜ ‖x̂ₜ - x₂‖₂²

    the masked version is defined as

    .. math:: \𝗆𝗌𝖾(x̂，x) ≔ \tfrac{1}{∑ₜ⋁ₖmₜₖ} ∑ₜ ‖mₜ⊙(x̂ₜ - xₜ)‖₂²

    where $m ⊙ z ≔ \where(m, z, 0)$ denotes elementwise masked selection. Thus,
    masked ``NaN`` and infinite entries do not contribute; in particular, $0 ⊙ 𝙽𝚊𝙽 = 0$.
    """

    penalty = staticmethod(mse)
    reduction = Reduction.MEAN


class RMSE_Loss(SeparableLoss):
    r"""Root mean square error.

    .. math:: \𝗋𝗆𝗌𝖾(x̂，x) ≔ 𝔼[ \sqrt{\tfrac{1}{T} ∑ₜ ‖x̂ₜ - x₂‖₂²} ]

    the masked version is defined as

    .. math:: \𝗋𝗆𝗌𝖾(x̂，x) ≔ 𝔼[ \sqrt{\tfrac{1}{∑ₜ⋁ₖmₜₖ} ∑ₜ ‖mₜ ⊙(x̂ₜ - xₜ)‖₂² } ]

    where $m ⊙ z ≔ \where(m, z, 0)$ denotes elementwise masked selection. Thus,
    masked ``NaN`` and infinite entries do not contribute; in particular, $0 ⊙ 𝙽𝚊𝙽 = 0$.

    Args:
        time_dim: Tensor axis containing the time steps.
        channel_dim: Tensor axes containing the channels; use ``()`` for a
            univariate time series.
        reduction:
    """

    penalty = staticmethod(rmse)
    reduction = Reduction.MEAN


class NRMSE_Loss(SeparableLoss):
    r"""Compute the normalized root mean square error.

    .. math:: \nrmse(x̂，x) ≔ \frac{ \sqrt{ T⁻¹ ∑ₜ‖x̂ₜ - xₜ‖² } }{ T⁻¹ ∑ₜ‖xₜ‖₁ }

    the masked version is defined as

    .. math:: \nrmse(x̂，x) ≔ \frac
        { \sqrt{ \tfrac{1}{∑ₛ⋁ₖmₛₖ} ∑ₜ‖mₜ⊙(x̂ₜ - xₜ)‖₂² } }
        { \tfrac{1}{∑ₛ⋁ₖmₛₖ} ∑ₜ‖mₜ⊙xₜ‖₁ }

    where $m ⊙ z ≔ \where(m, z, 0)$ denotes elementwise masked selection. Thus,
    masked ``NaN`` and infinite entries do not contribute; in particular, $0 ⊙ 𝙽𝚊𝙽 = 0$.

    Args:
        predictions: Predicted values of shape ``[..., N, *D]``.
        targets: Target values of shape ``[..., N, *D]``.
        time_dim: Tensor axis containing the time steps.
        channel_dim: Tensor axes containing the channels;
            use ``()`` for a univariate time series.
        mask: Boolean mask selecting values that contribute to the error.
            All selected values should be finite and not null.

    .. math:: 𝖭𝖱𝖬𝖲𝖤(x̂，x) ≔ \frac{ ‖x̂-x‖_{2⁎} }{ ‖x‖_{1⁎} }

    where $‖x‖_{p⁎} ≔ \sqrt[p]{ (1/T)∑ₜₖ|xₜₖ|ᵖ }$ is a scaled $p$-norm of the tensor.
    """

    penalty = staticmethod(nrmse)
    reduction = Reduction.MEAN


class ND_Loss(SeparableLoss):
    r"""Compute the normalized deviation.

    .. math:: \nd(x̂，x) ≔ \frac{ ∑ₜ‖x̂ₜ - xₜ‖₁ }{ ∑ₜ‖xₜ‖₁ } = \frac{‖x̂-x‖_{1,1⁎}}{‖x‖_{1,1⁎}}

    the masked version is defined as

    .. math:: \nd(x̂，x) ≔ \frac{ ∑ₜ‖mₜ⊙(x̂ₜ - xₜ)‖₁ }{ ∑ₜ‖mₜ⊙xₜ‖₁ }

    where $m ⊙ z ≔ \where(m, z, 0)$ denotes elementwise masked selection. Thus,
    masked ``NaN`` and infinite entries do not contribute; in particular, $0 ⊙ 𝙽𝚊𝙽 = 0$.

    Note:
        This loss is equivalent to `lp_loss` with ``p=1``, ``relative=True``,
        and ``normalization=observation_count``.

        .. math:: nd(x̂，x) ≔ \frac{‖x̂ - x‖_{1⁎}}{‖x‖_{1⁎}}

        where $‖x‖_{p⁎} ≔ \sqrt[p]{(1/T)∑ₜₖ|xₜₖ|ᵖ}$ is a scaled $p$-norm of the tensor.
    """

    penalty = staticmethod(nd)
    reduction = Reduction.MEAN


class LP_Loss(BaseSequenceLoss):
    r"""LP-loss for sequence spaces."""
