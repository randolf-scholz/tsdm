r"""Penalty functions for sequence data."""

__all__ = [
    "Normalization",
    "lp_norm",
    # error metrics
    "l2_error",
    "lp_error",
    "mae",
    "mse",
    "nd",
    "nrmse",
    "quantile_error",
    "rmse",
]


from enum import StrEnum
from math import prod
from typing import assert_never

import torch
from torch import Tensor


class Normalization(StrEnum):
    r"""Built-in divisors for powered sequence values.

    Let $T$ be the sequence length, $K$ the total number of channels, and $mₜₖ$
    the validity mask. With no mask, treat every entry as observed. The divisor
    $sₜₖ$ for each scheme is:

    - `none`: $sₜₖ=1$; retain the unnormalized values.
    - `sequence_length`: $sₜₖ=∑ₛ⋁ⱼmₛⱼ$, or $T$ without a mask.
        Give sequences equal total time weight regardless of their observed length.
    - `channel_prevalence`: $sₜₖ=∑ₛmₛₖ$, or $T$ without a mask.
        Normalize each channel by its observation count.
    - `timepoint_coverage`: $sₜₖ=(∑ₛ⋁ⱼmₛⱼ)(∑ⱼmₜⱼ)$, or $TK$ without a mask.
        Give each observed time point equal total weight across its observed channels.
    - `observation_count`: $sₜₖ=∑ₛⱼmₛⱼ$, or $T⋅K$ without a mask.
        Compute a mean over all observed values.

    Divisors computed from a mask are clamped to at least one. The reciprocal
    $γₜₖ=1/sₜₖ$ is the normalization factor used in sequence norms.
    """

    NONE = "none"
    SEQUENCE_LENGTH = "sequence_length"
    CHANNEL_PREVALENCE = "channel_prevalence"
    TIMEPOINT_COVERAGE = "timepoint_coverage"
    OBSERVATION_COUNT = "observation_count"

    def apply(
        self,
        values: Tensor,  # Float[..., $N, *D]
        /,
        *,
        time_dim: int,
        channel_dim: int | tuple[int, ...],
        mask: Tensor | None = None,  # Float[..., $N, *D]
    ) -> Tensor:  # Float[..., $N, *D]
        channel_dim = (channel_dim,) if isinstance(channel_dim, int) else channel_dim
        match self:
            case Normalization.NONE:
                return values

            case Normalization.SEQUENCE_LENGTH:
                normalizer = (
                    values.shape[time_dim]
                    if mask is None
                    else (
                        mask.any(dim=channel_dim, keepdim=True)
                        .sum(dim=time_dim, keepdim=True)
                        .clamp_min(1)
                    )
                )

            case Normalization.CHANNEL_PREVALENCE:
                normalizer = (
                    values.shape[time_dim]
                    if mask is None
                    else mask.sum(dim=time_dim, keepdim=True).clamp_min(1)
                )

            case Normalization.TIMEPOINT_COVERAGE:
                normalizer = (
                    values.shape[time_dim] * prod(values.shape[d] for d in channel_dim)
                    if mask is None
                    else (
                        mask.any(dim=channel_dim, keepdim=True)
                        .sum(dim=time_dim, keepdim=True)
                        .mul(mask.sum(dim=channel_dim, keepdim=True))
                        .clamp_min(1)
                    )
                )

            case Normalization.OBSERVATION_COUNT:
                normalizer = (
                    values.shape[time_dim] * prod(values.shape[d] for d in channel_dim)
                    if mask is None
                    else mask.sum(dim=(time_dim, *channel_dim), keepdim=True).clamp_min(
                        1
                    )
                )

            case other:
                assert_never(other)

        return values / normalizer


def _normalize_dims(
    *,
    ndim: int,
    time_dim: int,
    channel_dim: int | tuple[int, ...],
) -> tuple[int, tuple[int, ...]]:
    r"""Validate and normalize time and channel dimensions."""
    channel_dims = (channel_dim,) if isinstance(channel_dim, int) else channel_dim
    if not all(-ndim <= dim < ndim for dim in (time_dim, *channel_dims)):
        raise ValueError(f"channel or time dims are out of range for {ndim=}.")

    time_dim %= ndim
    channel_dims = tuple(dim % ndim for dim in channel_dims)
    if ndim < 1:
        raise ValueError("x must have at least a time dimension.")
    if time_dim in channel_dims:
        raise ValueError("time_dim and channel_dim must be disjoint.")
    if len(set(channel_dims)) != len(channel_dims):
        raise ValueError("channel_dim must not contain duplicate dimensions.")
    if channel_dims != tuple(sorted(channel_dims)):
        raise ValueError("channel_dim must be strictly increasing.")

    return time_dim, channel_dims


def lp_norm(
    x: Tensor,  # Float[..., $N, *D]
    /,
    *,
    time_dim: int,
    channel_dim: int | tuple[int, ...],  # *D
    # optional args
    p: float = 2.0,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
    time_weights: Tensor | None = None,  # Float[..., $N]
    channel_weights: Tensor | None = None,  # Float[*D]
    normalization: Normalization | Tensor | None = Normalization.CHANNEL_PREVALENCE,
) -> Tensor:  # Float[...]
    r"""Compute a normalized entry-wise norm over time and channel axes.

    $$ ℓₚ^*(x) ≔ \Bigl(∑ₜ wₜ^{\sf time}∑ₖ wₖ^{\sf channel}γₜₖ(mₜₖ⊙|xₜₖ|ᵖ)\Bigr)^{1/p} $$

    Here $m⊙z$ denotes element-wise masked selection via `torch.where(m, z, 0)`,
    so masked NaNs and infinities do not contribute; in particular, $0⊙\nan=0$.
    Without a mask, every entry is selected, including NaNs.

    The factor $γₜₖ=1/sₜₖ$ is the reciprocal divisor specified by `Normalization`.
    Missing weights are one. Masked entries are excluded before taking powers.
    An entirely masked sequence returns zero, including for negative orders.

    Args:
        x: Tensor of shape `(..., $N, *D)`, with the specified time and channel axes.
        time_dim: Axis containing time steps, disjoint from the channel axes.
        channel_dim: Channel axes, strictly increasing after normalizing negative
            indices and without duplicates. Use `()` for a univariate series.
        p: Finite nonzero norm order. Zero and infinite orders raise
            `NotImplementedError`; NaN raises `ValueError`.
        mask: Boolean tensor of the same shape as `x`, selecting valid entries.
        time_weights: Nonnegative importance weights aligned with the trailing
            nonchannel axes, including the time axis for the usual data layout.
        channel_weights: Nonnegative importance weights, with one axis per channel axis.
        normalization: A `Normalization` scheme or its string value, applied before
            weighting and summation. `None` disables normalization. A custom tensor
            currently raises `NotImplementedError`.
    """
    assert mask is None or mask.shape == x.shape
    time_dim, ch_dims = _normalize_dims(
        ndim=x.ndim,
        time_dim=time_dim,
        channel_dim=channel_dim,
    )

    match p:
        case torch.inf:
            raise NotImplementedError
        case _ if p == -torch.inf:
            raise NotImplementedError
        case 0.0:
            raise NotImplementedError
        case _ if p > 0:  # p∈(0,∞)
            values = x if mask is None else torch.where(mask, x, 0.0)
            values = values.abs().pow(p)
        case _ if p < 0:  # p∈(-∞,0)
            values = x if mask is None else torch.where(mask, x, 1.0)
            values = values.abs().pow(p)
            if mask is not None:
                values = torch.where(mask, values, 0.0)
        case _:  # NAN
            raise ValueError("p must not be NaN.")

    match normalization:
        case None:
            pass
        case Tensor():
            raise NotImplementedError
        case name:
            scheme = Normalization(name)
            values = scheme.apply(
                values,
                mask=mask,
                time_dim=time_dim,
                channel_dim=ch_dims,
            )

    if (w_t := time_weights) is not None:
        time_axes = tuple(axis for axis in range(x.ndim) if axis not in ch_dims)
        if not 1 <= w_t.ndim <= len(time_axes):
            raise ValueError(f"Expected 1 <= {w_t.ndim=} <= {len(time_axes)=}")
        dims = time_axes[-w_t.ndim :]
        w_t = w_t[*(slice(None) if d in dims else None for d in range(x.ndim))]
        values = w_t * values

    if (w_k := channel_weights) is not None:
        if w_k.ndim != len(ch_dims):
            raise ValueError(f"Expected {w_k.ndim=} to equal {len(ch_dims)=}")
        w_k = w_k[*(slice(None) if d in ch_dims else None for d in range(x.ndim))]
        values = w_k * values

    reduced = values.sum(dim=(time_dim, *ch_dims))
    if mask is None:
        return reduced.pow(1 / p)

    observed = mask.any(dim=(time_dim, *ch_dims))
    safe_reduced = torch.where(observed, reduced, 1.0)
    return torch.where(observed, safe_reduced.pow(1 / p), 0.0)


def lp_error(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    time_dim: int,
    channel_dim: int | tuple[int, ...],  # *D
    # optional args
    p: float = 2.0,
    relative: bool = False,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
    normalization: Normalization | Tensor | None = Normalization.CHANNEL_PREVALENCE,
    # weights:
    time_weights: Tensor | None = None,  # Float[..., $N]
    channel_weights: Tensor | None = None,  # Float[*D]
) -> Tensor:  # Float
    norms = lp_norm(
        predictions - targets,
        p=p,
        mask=mask,
        channel_dim=channel_dim,
        channel_weights=channel_weights,
        time_dim=time_dim,
        time_weights=time_weights,
        normalization=normalization,
    )
    if relative:
        norms = norms / lp_norm(
            targets,
            p=p,
            mask=mask,
            channel_dim=channel_dim,
            channel_weights=channel_weights,
            time_dim=time_dim,
            time_weights=time_weights,
            normalization=normalization,
        )
    return norms


def quantile_error(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    q: float,
    # optional args
    time_dim: int = -2,
    channel_dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
) -> Tensor:  # Float[...]
    r"""Compute a time-averaged pinball penalty, summed over channels.

    $$ ℓ_q(x̂，x) ≔ Tₘ⁻¹∑ₜ∑ₖ mₜₖ⊙ρ_q(x̂ₜₖ-xₜₖ) $$

    Here $m⊙z$ denotes element-wise masked selection via `torch.where(m, z, 0)`,
    so masked NaNs and infinities do not contribute; in particular, $0⊙\nan=0$.
    Without a mask, every entry is selected, including NaNs.

    For residuals $rₜₖ=x̂ₜₖ-xₜₖ$, define $ρ_q(r)=\max((q-1)r，qr)$.
    $Tₘ=\max(1,∑ₜ⋁ₖmₜₖ)$ counts time steps with an observed channel, clamped to one.
    Without a mask, $Tₘ=T$, where $T$ is the size of the `time_dim` axis.

    For nonnegative residuals the coefficient is $q$; for negative residuals it is
    $1-q$. Thus, with this residual sign, the optimal constant prediction targets
    quantile $1-q$. Supply $q∈[0,1]$; this function does not validate that interval.

    Predictions, targets, and any mask must have the same shape. An entirely
    masked sequence produces zero. Use `channel_dim=()` for a univariate sequence.
    """
    assert predictions.shape == targets.shape
    assert mask is None or mask.shape == predictions.shape
    time_dim, channel_dims = _normalize_dims(
        ndim=predictions.ndim,
        time_dim=time_dim,
        channel_dim=channel_dim,
    )
    reduced_time_dim = time_dim - sum(dim < time_dim for dim in channel_dims)

    r = predictions - targets
    if mask is not None:
        r = torch.where(mask, r, 0.0)
    errors = torch.maximum((q - 1) * r, q * r)

    # aggregate over channels and time
    # FIXME: https://github.com/pytorch/pytorch/issues/29137
    errors = errors if not channel_dims else errors.sum(dim=channel_dims)
    errors = errors.sum(dim=reduced_time_dim)

    if mask is None:
        return errors.div(predictions.shape[time_dim])

    time_mask = mask if not channel_dims else mask.any(dim=channel_dims)
    observed_timepoints = time_mask.sum(dim=reduced_time_dim).clamp_min(1)
    return errors.div(observed_timepoints)


def mae(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    # optional args
    time_dim: int = -2,
    channel_dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
) -> Tensor:  # Float[...]
    r"""Mean absolute error for each sequence pair.

    $$ \mae(x̂，x) ≔ Tₘ⁻¹∑ₜ ‖mₜ⊙(x̂ₜ-xₜ)‖₁ $$

    Here $m⊙z$ denotes element-wise masked selection via `torch.where(m, z, 0)`,
    so masked NaNs and infinities do not contribute; in particular, $0⊙\nan=0$.
    Without a mask, every entry is selected, including NaNs.

    $Tₘ=\max(1,∑ₜ⋁ₖmₜₖ)$ counts time steps with an observed channel, clamped to one.
    Without a mask, $Tₘ=T$, where $T$ is the size of the `time_dim` axis.

    Channel norms sum across channel axes; averaging is over observed time steps.
    Selected values must be finite; an entirely masked sequence produces zero.

    Args:
        predictions: Predicted values of shape `(..., $N, *D)`.
        targets: Target values of the same shape as `predictions`.
        time_dim: Tensor axis containing time steps.
        channel_dim: Tensor axes containing channels; use `()` for univariate data.
        mask: Boolean tensor of the same shape as the data, selecting valid entries.
    """
    assert predictions.shape == targets.shape
    assert mask is None or mask.shape == predictions.shape
    time_dim, channel_dims = _normalize_dims(
        ndim=predictions.ndim,
        time_dim=time_dim,
        channel_dim=channel_dim,
    )
    reduced_time_dim = time_dim - sum(dim < time_dim for dim in channel_dims)

    residuals = predictions - targets
    if mask is not None:
        residuals = torch.where(mask, residuals, 0.0)
    errors = residuals.abs()

    # aggregate over channels and time
    # FIXME: https://github.com/pytorch/pytorch/issues/29137
    errors = errors if not channel_dims else errors.sum(dim=channel_dims)
    errors = errors.sum(dim=reduced_time_dim)

    if mask is None:
        return errors.div(predictions.shape[time_dim])

    time_mask = mask if not channel_dims else mask.any(dim=channel_dims)
    observed_timepoints = time_mask.sum(dim=reduced_time_dim).clamp_min(1)
    return errors.div(observed_timepoints)


def l2_error(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    # optional args
    time_dim: int = -2,
    channel_dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
) -> Tensor:  # Float[...]
    r"""Mean Euclidean error for each sequence pair.

    $$ ℓ₂(x̂，x) ≔ Tₘ⁻¹∑ₜ ‖mₜ⊙(x̂ₜ-xₜ)‖₂ $$

    Here $m⊙z$ denotes element-wise masked selection via `torch.where(m, z, 0)`,
    so masked NaNs and infinities do not contribute; in particular, $0⊙\nan=0$.
    Without a mask, every entry is selected, including NaNs.

    $Tₘ=\max(1,∑ₜ⋁ₖmₜₖ)$ counts time steps with an observed channel, clamped to one.
    Without a mask, $Tₘ=T$, where $T$ is the size of the `time_dim` axis.

    Channel norms sum across channel axes; averaging is over observed time steps.
    Selected values must be finite; an entirely masked sequence produces zero.

    Args:
        predictions: Predicted values of shape `(..., $N, *D)`.
        targets: Target values of the same shape as `predictions`.
        time_dim: Tensor axis containing time steps.
        channel_dim: Tensor axes containing channels; use `()` for univariate data.
        mask: Boolean tensor of the same shape as the data, selecting valid entries.
    """
    assert predictions.shape == targets.shape
    assert mask is None or mask.shape == predictions.shape
    time_dim, channel_dims = _normalize_dims(
        ndim=predictions.ndim,
        time_dim=time_dim,
        channel_dim=channel_dim,
    )
    reduced_time_dim = time_dim - sum(d < time_dim for d in channel_dims)

    residuals = predictions - targets
    if mask is not None:
        residuals = torch.where(mask, residuals, 0.0)
    errors = residuals.square()

    # aggregate over channels and time
    # FIXME: https://github.com/pytorch/pytorch/issues/29137
    errors = errors if not channel_dims else errors.sum(dim=channel_dims)
    errors = errors.sqrt()
    errors = errors.sum(dim=reduced_time_dim)

    if mask is None:
        return errors.div(predictions.shape[time_dim])

    time_mask = mask if not channel_dims else mask.any(dim=channel_dims)
    observed_timepoints = time_mask.sum(dim=reduced_time_dim).clamp_min(1)
    return errors.div(observed_timepoints)


def mse(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    # optional args
    time_dim: int = -2,
    channel_dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
) -> Tensor:  # Float[...]
    r"""Mean squared error for each sequence pair.

    $$ \mse(x̂，x) ≔ Tₘ⁻¹∑ₜ ‖mₜ⊙(x̂ₜ-xₜ)‖₂² $$

    Here $m⊙z$ denotes element-wise masked selection via `torch.where(m, z, 0)`,
    so masked NaNs and infinities do not contribute; in particular, $0⊙\nan=0$.
    Without a mask, every entry is selected, including NaNs.

    $Tₘ=\max(1,∑ₜ⋁ₖmₜₖ)$ counts time steps with an observed channel, clamped to one.
    Without a mask, $Tₘ=T$, where $T$ is the size of the `time_dim` axis.

    Channel norms sum across channel axes; averaging is over observed time steps.
    Selected values must be finite; an entirely masked sequence produces zero.

    Args:
        predictions: Predicted values of shape `(..., $N, *D)`.
        targets: Target values of the same shape as `predictions`.
        time_dim: Tensor axis containing time steps.
        channel_dim: Tensor axes containing channels; use `()` for univariate data.
        mask: Boolean tensor of the same shape as the data, selecting valid entries.
    """
    assert predictions.shape == targets.shape
    assert mask is None or mask.shape == predictions.shape
    return lp_norm(
        predictions - targets,
        p=2.0,
        mask=mask,
        time_dim=time_dim,
        channel_dim=channel_dim,
        normalization=Normalization.SEQUENCE_LENGTH,
    ).square()


def rmse(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    # optional args
    time_dim: int = -2,
    channel_dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
) -> Tensor:  # Float[...]
    r"""Root mean squared error for each sequence pair.

    $$ \op{RMSE}(x̂，x) ≔ \sqrt{Tₘ⁻¹∑ₜ ‖mₜ⊙(x̂ₜ-xₜ)‖₂²} $$

    Here $m⊙z$ denotes element-wise masked selection via `torch.where(m, z, 0)`,
    so masked NaNs and infinities do not contribute; in particular, $0⊙\nan=0$.
    Without a mask, every entry is selected, including NaNs.

    $Tₘ=\max(1,∑ₜ⋁ₖmₜₖ)$ counts time steps with an observed channel, clamped to one.
    Without a mask, $Tₘ=T$, where $T$ is the size of the `time_dim` axis.

    Channel norms sum across channel axes; averaging is over observed time steps.
    Selected values must be finite; an entirely masked sequence produces zero.

    Args:
        predictions: Predicted values of shape `(..., $N, *D)`.
        targets: Target values of the same shape as `predictions`.
        time_dim: Tensor axis containing time steps.
        channel_dim: Tensor axes containing channels; use `()` for univariate data.
        mask: Boolean tensor of the same shape as the data, selecting valid entries.
    """
    assert predictions.shape == targets.shape
    assert mask is None or mask.shape == predictions.shape
    time_dim, channel_dims = _normalize_dims(
        ndim=predictions.ndim,
        time_dim=time_dim,
        channel_dim=channel_dim,
    )
    reduced_time_dim = time_dim - sum(dim < time_dim for dim in channel_dims)

    residuals = predictions - targets
    if mask is not None:
        residuals = torch.where(mask, residuals, 0.0)
    errors = residuals.square()

    # aggregate over channels and time
    # FIXME: https://github.com/pytorch/pytorch/issues/29137
    errors = errors if not channel_dims else errors.sum(dim=channel_dims)
    errors = errors.sum(dim=reduced_time_dim)

    if mask is None:
        return errors.div(predictions.shape[time_dim]).sqrt()

    time_mask = mask if not channel_dims else mask.any(dim=channel_dims)
    observed_timepoints = time_mask.sum(dim=reduced_time_dim).clamp_min(1)
    return errors.div(observed_timepoints).sqrt()


def nd(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    # optional args
    time_dim: int = -2,
    channel_dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
) -> Tensor:  # Float[...]
    r"""Normalized deviation for each sequence pair.

    $$ \nd(x̂，x) ≔ \frac{∑ₜ ‖mₜ⊙(x̂ₜ-xₜ)‖₁}{∑ₜ ‖mₜ⊙xₜ‖₁} $$

    Here $m⊙z$ denotes element-wise masked selection via `torch.where(m, z, 0)`,
    so masked NaNs and infinities do not contribute; in particular, $0⊙\nan=0$.
    Without a mask, every entry is selected, including NaNs.

    The ratio normalizes total absolute residuals by total absolute target values.
    Selected values must be finite.

    A zero target norm can produce NaN or infinity. An entirely masked
    sequence produces NaN because both norms are zero.

    This penalty equals `lp_error` with `p=1`, `relative=True`, and
    `normalization=Normalization.OBSERVATION_COUNT`.

    Args:
        predictions: Predicted values of shape `(..., $N, *D)`.
        targets: Target values of the same shape as `predictions`.
        time_dim: Tensor axis containing time steps.
        channel_dim: Tensor axes containing channels; use `()` for univariate data.
        mask: Boolean tensor of the same shape as the data, selecting valid entries.
    """
    assert predictions.shape == targets.shape
    assert mask is None or mask.shape == predictions.shape
    residual_norm = lp_norm(
        predictions - targets,
        p=1.0,
        mask=mask,
        time_dim=time_dim,
        channel_dim=channel_dim,
        normalization=Normalization.OBSERVATION_COUNT,
    )
    target_norm = lp_norm(
        targets,
        p=1.0,
        mask=mask,
        time_dim=time_dim,
        channel_dim=channel_dim,
        normalization=Normalization.OBSERVATION_COUNT,
    )
    return residual_norm / target_norm


def nrmse(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    # optional args
    time_dim: int = -2,
    channel_dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
) -> Tensor:  # Float[...]
    r"""Normalized root mean squared error for each sequence pair.

    $$ \nrmse(x̂，x) ≔ \frac{\sqrt{Tₘ⁻¹∑ₜ ‖mₜ⊙(x̂ₜ-xₜ)‖₂²}}{Tₘ⁻¹∑ₜ ‖mₜ⊙xₜ‖₁} $$

    Here $m⊙z$ denotes element-wise masked selection via `torch.where(m, z, 0)`,
    so masked NaNs and infinities do not contribute; in particular, $0⊙\nan=0$.
    Without a mask, every entry is selected, including NaNs.

    $Tₘ=\max(1,∑ₜ⋁ₖmₜₖ)$ counts time steps with an observed channel, clamped to one.
    Without a mask, $Tₘ=T$, where $T$ is the size of the `time_dim` axis.

    Channel norms sum across the channel axes; averaging uses the number of
    observed time steps. Selected values must be finite.

    A zero target norm can produce NaN or infinity. An entirely masked
    sequence produces NaN because both norms are zero.

    Args:
        predictions: Predicted values of shape `(..., $N, *D)`.
        targets: Target values of the same shape as `predictions`.
        time_dim: Tensor axis containing time steps.
        channel_dim: Tensor axes containing channels; use `()` for univariate data.
        mask: Boolean tensor of the same shape as the data, selecting valid entries.
    """
    assert predictions.shape == targets.shape
    assert mask is None or mask.shape == predictions.shape
    residual_norm = lp_norm(
        predictions - targets,
        p=2.0,
        mask=mask,
        time_dim=time_dim,
        channel_dim=channel_dim,
        normalization=Normalization.SEQUENCE_LENGTH,
    )
    target_norm = lp_norm(
        targets,
        p=1.0,
        mask=mask,
        time_dim=time_dim,
        channel_dim=channel_dim,
        normalization=Normalization.SEQUENCE_LENGTH,
    )
    return residual_norm / target_norm
