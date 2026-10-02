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
    "rmse",
]


from enum import StrEnum
from math import prod
from typing import assert_never

import torch
from torch import Tensor


class Normalization(StrEnum):
    r"""Built-in normalization schemes for sequential norms.

    - ``sequence_length``: Divide by the number of time steps with at least one observed channel

            $sₜₖ ≔ T$ if no mask else $sₜₖ ≔ ∑ₛ⋁ⱼmₛⱼ$.

      Use it to give padded sequences equal weight regardless of their length.

    - ``channel_prevalence``: Divide each channel by its number of observations

            $sₜₖ ≔ T$ if no mask else $sₜₖ ≔ ∑ₛmₛₖ$.

      Use it to prevent frequently observed channels from dominating sparse channels.

    - ``timepoint_coverage``: Divide by both sequence length and the number of observed channels at each time step

            $sₜₖ ≔ T⋅K$ if no mask, else $sₜₖ ≔ (∑ₛ⋁ⱼmₛⱼ)⋅(∑ⱼmₜⱼ)$

      Use it to give each observed time point equal weight.

    - ``observation_count``: Divide by the total number of observed values

            $sₜₖ ≔ T⋅K$ if no mask, else $sₜₖ ≔ ∑ₛⱼmₛⱼ$.

      Use it to compute a global mean over observations.
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
    r"""Compute time-normalized lp-norm for sequence space $\Seq(Y)$.

    .. math:: ℓₚ^*(x) = \Bigl( ∑ₜ wₜᵗⁱᵐᵉ ∑ₖ wₖᶜʰ γₜₖ (mₜₖ⊙|xₜₖ|ᵖ) \Bigr)^{1/p}

    where $m ⊙ z ≔ \where(m, z, 0)$ denotes elementwise masked selection. Thus,
    masked ``NaN`` and infinite entries do not contribute; in particular, $0 ⊙ 𝙽𝚊𝙽 = 0$.

    Args:
        x: Tensor of shape ``[..., $N, *D]``.
        p: Order of the norm. Note: $p∈\{0, ±∞\}$ are currently not implemented.
        mask: Boolean mask indicating valid values.
        time_dim: Required tensor axis of the time dimension.
        time_weights: Importance weight $wₜᵗⁱᵐᵉ$ for each time step.
        channel_dim: Required tensor axes of the channel dimensions. Use ``()``
            for an univariate time series.
        channel_weights: Importance weight $wₖᶜʰ$ for each channel.
        normalization: Normalization applied to the powered values. ``None``
            performs no normalization ($sₜₖ=1$). The built-in schemes are:

            - ``none``: $γₜₖ = 1$.
            - ``sequence_length``: Divide by the number of time steps with at
              least one observed channel $sₜₖ ≔ T$ if no mask else $sₜₖ ≔ ∑ₛ⋁ⱼmₛⱼ$.
              Use it to give padded sequences equal weight regardless of their length.
            - ``channel_prevalence``: Divide each channel by its number of observations
              $sₜₖ ≔ T$ if no mask else $sₜₖ ≔ ∑ₛmₛₖ$.
              Use it to prevent frequently observed channels from dominating sparse channels.
            - ``timepoint_coverage``: Divide by both sequence length and the
              number of observed channels at each time step
              $sₜₖ ≔ T⋅K$ if no mask, else $sₜₖ ≔ (∑ₛ⋁ⱼmₛⱼ)⋅(∑ⱼmₜⱼ)$.
              Use it to give each observed time point equal weight.
            - ``observation_count``: Divide by the total number of observed
              values $sₜₖ ≔ T⋅K$ if no mask, else $sₜₖ ≔ ∑ₛⱼmₛⱼ$.
              Use it to compute a global mean over observations.

            A custom tensor is reserved for a future user-defined normalization
            scheme and currently raises ``NotImplementedError``.
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
) -> Tensor:  # Float[()]
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


def mae(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    # optional args
    time_dim: int = -2,
    channel_dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
) -> Tensor:  # Float[...]
    r"""Mean absolute error.

    .. math:: \mae(x̂，x) ≔ \tfrac{1}{T} ∑ₜ ‖x̂ₜ - xₜ‖₁ = ‖x̂ - x‖_{1,1⁎}

    the masked version is defined as

    .. math:: \mae(x̂，x) ≔ \tfrac{1}{∑ₜ⋁ₖmₜₖ} ∑ₜ ‖mₜ⊙(x̂ₜ - xₜ)‖₁

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
    """
    assert predictions.shape == targets.shape
    assert mask is None or mask.shape == predictions.shape
    time_dim, channel_dims = _normalize_dims(
        ndim=predictions.ndim,
        time_dim=time_dim,
        channel_dim=channel_dim,
    )

    residuals = predictions - targets
    if mask is not None:
        residuals = torch.where(mask, residuals, 0.0)
    absolute_errors = residuals.abs()

    channel_absolute_errors = (
        absolute_errors if not channel_dims else absolute_errors.sum(dim=channel_dims)
    )
    reduced_time_dim = time_dim - sum(dim < time_dim for dim in channel_dims)
    time_absolute_errors = channel_absolute_errors.sum(dim=reduced_time_dim)

    if mask is None:
        return time_absolute_errors.div(predictions.shape[time_dim])

    time_mask = mask if not channel_dims else mask.any(dim=channel_dims)
    observed_timepoints = time_mask.sum(dim=reduced_time_dim).clamp_min(1)
    return time_absolute_errors.div(observed_timepoints)


def l2_error(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    # optional args
    time_dim: int = -2,
    channel_dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
) -> Tensor:  # Float[...]
    r"""Mean euclidean error.

    .. math:: ℓ(x̂，x) ≔ \tfrac{1}{T} ∑ₜ ‖x̂ₜ - x₂‖₂ = ‖x̂ - x‖_{2,1⁎}

    the masked version is defined as

    .. math:: ℓ(x̂，x) ≔ \tfrac{1}{∑ₛ⋁ₖmₛₖ} ∑ₜ‖mₜ⊙(x̂ₜ - xₜ)‖₂

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
    """
    assert predictions.shape == targets.shape
    assert mask is None or mask.shape == predictions.shape
    time_dim, channel_dims = _normalize_dims(
        ndim=predictions.ndim,
        time_dim=time_dim,
        channel_dim=channel_dim,
    )

    residuals = predictions - targets
    if mask is not None:
        residuals = torch.where(mask, residuals, 0.0)
    squared_errors = residuals.square()

    channel_squared_errors = (
        squared_errors if not channel_dims else squared_errors.sum(dim=channel_dims)
    )
    reduced_time_dim = time_dim - sum(dim < time_dim for dim in channel_dims)
    time_euclidean_errors = channel_squared_errors.sqrt().sum(dim=reduced_time_dim)

    if mask is None:
        return time_euclidean_errors.div(predictions.shape[time_dim])

    time_mask = mask if not channel_dims else mask.any(dim=channel_dims)
    observed_timepoints = time_mask.sum(dim=reduced_time_dim).clamp_min(1)
    return time_euclidean_errors.div(observed_timepoints)


def mse(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    # optional args
    time_dim: int = -2,
    channel_dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
) -> Tensor:  # Float[...]
    r"""Mean square error.

    .. math:: \𝗆𝗌𝖾(x̂，x) ≔ \tfrac{1}{T} ∑ₜ ‖x̂ₜ - xₜ‖₂² = ‖x̂ - x‖²_{2,2⁎}

    the masked version is defined as

    .. math:: \𝗆𝗌𝖾(x̂，x) ≔ \tfrac{1}{∑ₜ⋁ₖmₜₖ} ∑ₜ ‖mₜ⊙(x̂ₜ - xₜ)‖₂²

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
    r"""Root mean square error.

    .. math:: \𝗋𝗆𝗌𝖾(x̂，x) ≔ \sqrt{\tfrac{1}{T} ∑ₜ ‖x̂ₜ - x₂‖₂²} = ‖x̂ - x‖_{2,2⁎}

    the masked version is defined as

    .. math:: \𝗋𝗆𝗌𝖾(x̂，x) ≔ \sqrt{\tfrac{1}{∑ₜ⋁ₖmₜₖ} ∑ₜ ‖mₜ ⊙(x̂ₜ - xₜ)‖₂² }

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
    """
    assert predictions.shape == targets.shape
    assert mask is None or mask.shape == predictions.shape
    time_dim, channel_dims = _normalize_dims(
        ndim=predictions.ndim,
        time_dim=time_dim,
        channel_dim=channel_dim,
    )

    residuals = predictions - targets
    if mask is not None:
        residuals = torch.where(mask, residuals, 0.0)
    squared_errors = residuals.square()

    channel_squared_errors = (
        squared_errors if not channel_dims else squared_errors.sum(dim=channel_dims)
    )
    reduced_time_dim = time_dim - sum(dim < time_dim for dim in channel_dims)
    time_squared_errors = channel_squared_errors.sum(dim=reduced_time_dim)

    if mask is None:
        return time_squared_errors.div(predictions.shape[time_dim]).sqrt()

    time_mask = mask if not channel_dims else mask.any(dim=channel_dims)
    observed_timepoints = time_mask.sum(dim=reduced_time_dim).clamp_min(1)
    return time_squared_errors.div(observed_timepoints).sqrt()


def nd(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    # optional args
    time_dim: int = -2,
    channel_dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
) -> Tensor:  # Float[...]
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
