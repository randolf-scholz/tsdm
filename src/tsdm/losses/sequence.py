r"""Loss functions for masked, variable-length sequences.

Sequences are stored in dense tensors with explicit time and channel axes.
Use a boolean mask to exclude padding and missing observations. Predictions
must be valid wherever the mask selects a target; NaNs are not masked
automatically.

Observation counts can vary substantially between channels. The general
`lp_loss` defaults to channel-prevalence normalization so that frequently
observed channels do not dominate solely because they have more observations.
Other normalization schemes are described by `Normalization`.

Sequence penalties produce one value per sample. `Reduction` then combines
those values, optionally using sample weights.
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

from functools import partial

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
    r"""Reduce normalized sequence residual norms over the batch dimensions.

    $$ 𝓛ₚ(x̂，x) ≔ Ψ\bigl((ℓₚ^*(x̂ₙ-xₙ))ₙ\bigr) $$

    Here $ℓₚ^*$ is the normalized sequence norm and $Ψ$ is the outer sample
    reduction. With `relative=True`, each penalty is divided by $ℓₚ^*(xₙ)$
    before applying $Ψ$.

    Compute per-sequence errors with `tsdm.losses.penalties.sequence.lp_error`,
    then apply `Reduction` with `sample_weights`. The default reduction is a mean.
    The norm defaults to channel-prevalence normalization; with `relative=True`,
    divide each residual norm by the corresponding target norm. A zero target norm
    can produce NaN or infinity.

    Args:
        predictions: Predicted sequences of shape `(..., $N, *D)`.
        targets: Target sequences of the same shape as `predictions`.
        time_dim: Tensor axis containing time steps.
        channel_dim: Tensor axes containing channels; use `()` for univariate data.
        p: Finite nonzero norm order.
        relative: Whether to divide by the corresponding target norm.
        mask: Boolean tensor selecting valid observations; NaNs are not inferred.
        reduction: Aggregation applied to the per-sequence errors.
        normalization: Normalization applied to the powered values within each norm.
        time_weights: Optional importance weights for time steps.
        channel_weights: Optional importance weights for channels.
        sample_weights: Optional weights for the outer sample reduction.
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
    r"""Base class for separable losses with a reduction over samples.

    $$ 𝓛(x̂,x) = \op{Reduce}\bigl((ℓ(x̂ₙ,xₙ))ₙ\bigr) $$

    Apply a penalty independently to each prediction-target pair, then combine the
    resulting values using `Reduction`, optionally with sample weights.

    The penalty operates within a sample; the outer reduction combines samples.
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
    r"""Mean Euclidean error for each sequence pair.

    $$ ℓ₂(x̂，x) ≔ Tₘ⁻¹∑ₜ ‖mₜ⊙(x̂ₜ-xₜ)‖₂ $$

    Here $m⊙z$ denotes element-wise masked selection via `torch.where(m, z, 0)`,
    so masked NaNs and infinities do not contribute; in particular, $0⊙\nan=0$.
    Without a mask, every entry is selected, including NaNs.

    $Tₘ=\max(1,∑ₜ⋁ₖmₜₖ)$ counts time steps with an observed channel, clamped to one.
    Without a mask, $Tₘ=T$, where $T$ is the size of the `time_dim` axis.

    Channel norms sum across channel axes; averaging is over observed time steps.
    Selected values must be finite; an entirely masked sequence produces zero.

    The class applies `Reduction.MEAN` to these per-sequence penalties. Optional
    `sample_weights` passed to `forward` give a weighted mean over the samples.

    Args:
        time_dim: Tensor axis containing time steps.
        channel_dim: Tensor axes containing channels; use `()` for univariate data.
    """

    penalty = staticmethod(l2_error)
    reduction = Reduction.MEAN


class MAE_Loss(SeparableLoss):
    r"""Mean absolute error for each sequence pair.

    $$ \mae(x̂，x) ≔ Tₘ⁻¹∑ₜ ‖mₜ⊙(x̂ₜ-xₜ)‖₁ $$

    Here $m⊙z$ denotes element-wise masked selection via `torch.where(m, z, 0)`,
    so masked NaNs and infinities do not contribute; in particular, $0⊙\nan=0$.
    Without a mask, every entry is selected, including NaNs.

    $Tₘ=\max(1,∑ₜ⋁ₖmₜₖ)$ counts time steps with an observed channel, clamped to one.
    Without a mask, $Tₘ=T$, where $T$ is the size of the `time_dim` axis.

    Channel norms sum across channel axes; averaging is over observed time steps.
    Selected values must be finite; an entirely masked sequence produces zero.

    The class applies `Reduction.MEAN` to these per-sequence penalties. Optional
    `sample_weights` passed to `forward` give a weighted mean over the samples.

    Args:
        time_dim: Tensor axis containing time steps.
        channel_dim: Tensor axes containing channels; use `()` for univariate data.
    """

    penalty = staticmethod(mae)
    reduction = Reduction.MEAN


class MSE_Loss(SeparableLoss):
    r"""Mean squared error for each sequence pair.

    $$ \mse(x̂，x) ≔ Tₘ⁻¹∑ₜ ‖mₜ⊙(x̂ₜ-xₜ)‖₂² $$

    Here $m⊙z$ denotes element-wise masked selection via `torch.where(m, z, 0)`,
    so masked NaNs and infinities do not contribute; in particular, $0⊙\nan=0$.
    Without a mask, every entry is selected, including NaNs.

    $Tₘ=\max(1,∑ₜ⋁ₖmₜₖ)$ counts time steps with an observed channel, clamped to one.
    Without a mask, $Tₘ=T$, where $T$ is the size of the `time_dim` axis.

    Channel norms sum across channel axes; averaging is over observed time steps.
    Selected values must be finite; an entirely masked sequence produces zero.

    The class applies `Reduction.MEAN` to these per-sequence penalties. Optional
    `sample_weights` passed to `forward` give a weighted mean over the samples.

    Args:
        time_dim: Tensor axis containing time steps.
        channel_dim: Tensor axes containing channels; use `()` for univariate data.
    """

    penalty = staticmethod(mse)
    reduction = Reduction.MEAN


class RMSE_Loss(SeparableLoss):
    r"""Root mean squared error for each sequence pair.

    $$ \op{RMSE}(x̂，x) ≔ \sqrt{Tₘ⁻¹∑ₜ ‖mₜ⊙(x̂ₜ-xₜ)‖₂²} $$

    Here $m⊙z$ denotes element-wise masked selection via `torch.where(m, z, 0)`,
    so masked NaNs and infinities do not contribute; in particular, $0⊙\nan=0$.
    Without a mask, every entry is selected, including NaNs.

    $Tₘ=\max(1,∑ₜ⋁ₖmₜₖ)$ counts time steps with an observed channel, clamped to one.
    Without a mask, $Tₘ=T$, where $T$ is the size of the `time_dim` axis.

    Channel norms sum across channel axes; averaging is over observed time steps.
    Selected values must be finite; an entirely masked sequence produces zero.

    The class applies `Reduction.MEAN` to these per-sequence penalties. Optional
    `sample_weights` passed to `forward` give a weighted mean over the samples.

    The square root is taken within each sequence before the sample mean.

    Args:
        time_dim: Tensor axis containing time steps.
        channel_dim: Tensor axes containing channels; use `()` for univariate data.
    """

    penalty = staticmethod(rmse)
    reduction = Reduction.MEAN


class NRMSE_Loss(SeparableLoss):
    r"""Normalized root mean squared error for each sequence pair.

    $$ \nrmse(x̂，x) ≔ \frac{ \sqrt{Tₘ⁻¹∑ₜ‖mₜ⊙(x̂ₜ-xₜ)‖₂²} }{ Tₘ⁻¹∑ₜ‖mₜ⊙xₜ‖₁ } $$

    Here $m⊙z$ denotes element-wise masked selection via `torch.where(m, z, 0)`,
    so masked NaNs and infinities do not contribute; in particular, $0⊙\nan=0$.
    Without a mask, every entry is selected, including NaNs.

    $Tₘ=\max(1,∑ₜ⋁ₖmₜₖ)$ counts time steps with an observed channel, clamped to one.
    Without a mask, $Tₘ=T$, where $T$ is the size of the `time_dim` axis.

    Channel norms sum across the channel axes; averaging uses the number of
    observed time steps. Selected values must be finite.

    A zero target norm can produce NaN or infinity. An entirely masked
    sequence produces NaN because both norms are zero.

    The class applies `Reduction.MEAN` to these per-sequence penalties. Optional
    `sample_weights` passed to `forward` give a weighted mean over the samples.

    Args:
        time_dim: Tensor axis containing time steps.
        channel_dim: Tensor axes containing channels; use `()` for univariate data.
    """

    penalty = staticmethod(nrmse)
    reduction = Reduction.MEAN


class ND_Loss(SeparableLoss):
    r"""Normalized deviation for each sequence pair.

    $$ \nd(x̂，x) ≔ \frac{ ∑ₜ‖mₜ⊙(x̂ₜ-xₜ)‖₁ }{ ∑ₜ ‖mₜ⊙xₜ‖₁ } $$

    Here $m⊙z$ denotes element-wise masked selection via `torch.where(m, z, 0)`,
    so masked NaNs and infinities do not contribute; in particular, $0⊙\nan=0$.
    Without a mask, every entry is selected, including NaNs.

    The ratio normalizes total absolute residuals by total absolute target values.
    Selected values must be finite.

    A zero target norm can produce NaN or infinity. An entirely masked
    sequence produces NaN because both norms are zero.

    This penalty equals `lp_error` with `p=1`, `relative=True`, and
    `normalization=Normalization.OBSERVATION_COUNT`.

    The class applies `Reduction.MEAN` to these per-sequence penalties. Optional
    `sample_weights` passed to `forward` give a weighted mean over the samples.

    Args:
        time_dim: Tensor axis containing time steps.
        channel_dim: Tensor axes containing channels; use `()` for univariate data.
    """

    penalty = staticmethod(nd)
    reduction = Reduction.MEAN


class LP_Loss(SeparableLoss):
    r"""Lᵖ-loss for sequence spaces."""

    reduction = Reduction.MEAN

    def __init__(
        self,
        *,
        p: float,
        time_dim: int,
        channel_dim: int | tuple[int, ...],
        reduction: Reduction = Reduction.MEAN,
    ) -> None:
        super().__init__(
            time_dim=time_dim,
            channel_dim=channel_dim,
            reduction=reduction,
            penalty=partial(lp_error, p=p),
        )
