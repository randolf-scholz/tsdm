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

from .base import BaseSequenceLoss, WrappedLoss
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
from .reductions import Reduction


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


class L2_Loss(WrappedLoss):
    r"""Mean Euclidean error.

    .. math:: ℓ(x̂，x) ≔ \tfrac{1}{T} ∑ₜ ‖x̂ₜ - x₂‖₂ = ‖x̂ - x‖_{2,1⁎}

    the masked version is defined as

    .. math:: ℓ(x̂，x) ≔ \tfrac{1}{∑ₛ⋁ₖmₛₖ} ∑ₜ‖mₜ⊙(x̂ₜ - xₜ)‖₂

    where $m ⊙ z ≔ \where(m, z, 0)$ denotes elementwise masked selection. Thus,
    masked ``NaN`` and infinite entries do not contribute; in particular, $0 ⊙ 𝙽𝚊𝙽 = 0$.
    """

    def __init__(
        self,
        *,
        time_dim: int,
        channel_dim: int | tuple[int, ...],
        reduction: Reduction = Reduction.MEAN,
    ) -> None:
        super().__init__(
            time_dim=time_dim,
            channel_dim=channel_dim,
            reduction=reduction,
            metric=l2_error,
        )


class MAE_Loss(WrappedLoss):
    r"""Mean absolute error.

    .. math:: \mae(x̂，x) ≔ \tfrac{1}{T} ∑ₜ ‖x̂ₜ - x₂‖₁

    the masked version is defined as

    .. math:: \mae(x̂，x) ≔ \tfrac{1}{∑ₜ⋁ₖmₜₖ} ∑ₜ ‖mₜ⊙(x̂ₜ - xₜ)‖₁

    where $m ⊙ z ≔ \where(m, z, 0)$ denotes elementwise masked selection. Thus,
    masked ``NaN`` and infinite entries do not contribute; in particular, $0 ⊙ 𝙽𝚊𝙽 = 0$.
    """

    def __init__(
        self,
        *,
        time_dim: int,
        channel_dim: int | tuple[int, ...],
        reduction: Reduction = Reduction.MEAN,
    ) -> None:
        super().__init__(
            time_dim=time_dim,
            channel_dim=channel_dim,
            reduction=reduction,
            metric=mae,
        )


class MSE_Loss(WrappedLoss):
    r"""Mean square error.

    .. math:: \𝗆𝗌𝖾(x̂，x) ≔ \tfrac{1}{T} ∑ₜ ‖x̂ₜ - x₂‖₂²

    the masked version is defined as

    .. math:: \𝗆𝗌𝖾(x̂，x) ≔ \tfrac{1}{∑ₜ⋁ₖmₜₖ} ∑ₜ ‖mₜ⊙(x̂ₜ - xₜ)‖₂²

    where $m ⊙ z ≔ \where(m, z, 0)$ denotes elementwise masked selection. Thus,
    masked ``NaN`` and infinite entries do not contribute; in particular, $0 ⊙ 𝙽𝚊𝙽 = 0$.
    """

    def __init__(
        self,
        *,
        time_dim: int,
        channel_dim: int | tuple[int, ...],
        reduction: Reduction = Reduction.MEAN,
    ) -> None:
        super().__init__(
            time_dim=time_dim,
            channel_dim=channel_dim,
            reduction=reduction,
            metric=mse,
        )


class RMSE_Loss(WrappedLoss):
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

    def __init__(
        self,
        *,
        time_dim: int,
        channel_dim: int | tuple[int, ...],
        reduction: Reduction = Reduction.MEAN,
    ) -> None:
        super().__init__(
            time_dim=time_dim,
            channel_dim=channel_dim,
            reduction=reduction,
            metric=mse,
        )


class NRMSE_Loss(WrappedLoss):
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

    def __init__(
        self,
        *,
        time_dim: int,
        channel_dim: int | tuple[int, ...],
        reduction: Reduction = Reduction.MEAN,
    ) -> None:
        super().__init__(
            time_dim=time_dim,
            channel_dim=channel_dim,
            reduction=reduction,
            metric=nrmse,
        )


class ND_Loss(WrappedLoss):
    __doc__ = nd.__doc__

    def __init__(
        self,
        *,
        time_dim: int,
        channel_dim: int | tuple[int, ...],
        reduction: Reduction = Reduction.MEAN,
    ) -> None:
        super().__init__(
            time_dim=time_dim,
            channel_dim=channel_dim,
            reduction=reduction,
            metric=nd,
        )


class LP_Loss(BaseSequenceLoss):
    r"""LP-loss for sequence spaces."""
