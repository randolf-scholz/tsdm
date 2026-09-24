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
    "Normalization",
    "LP_Loss",
    "Quantile_Loss",
    "RMSE_Loss",
    "SequentialMSE",
    # losses
    "lp_loss",
    "quantile_loss",
    "rmse",
    # functions
    "lp_norm",
]

from collections.abc import Callable
from enum import StrEnum
from math import prod
from typing import Any, Protocol, assert_never, cast

import torch
from torch import Tensor

from .base import BaseSequenceLoss
from .penalties import quantile_error
from .reductions import Reduction


class TaggedCallable[**P, R](Protocol[P, R]):
    __tags__: dict[str, Any]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


class MapsToTagged(Protocol):
    def __call__[**P, R](self, fn: Callable[P, R], /) -> TaggedCallable[P, R]: ...


def tags(**kwargs: object) -> MapsToTagged:
    def decorator[**P, R](fn: Callable[P, R], /) -> TaggedCallable[P, R]:
        fn.__tags__ = kwargs
        return cast("TaggedCallable[P, R]", fn)

    return decorator


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
    return Reduction(reduction).apply(norms, weights=sample_weights)


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

    .. math:: mae(x̂，x) ≔ \tfrac{1}{T} ∑ₜ ‖x̂ₜ - xₜ‖₁ = ‖x̂ - x‖_{1,1⁎}

    the masked version is defined as

    .. math:: mae(x̂，x) ≔ \tfrac{1}{∑ₜ⋁ₖmₜₖ} ∑ₜ ‖mₜ⊙(x̂ₜ - xₜ)‖₁

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


def mee(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    # optional args
    time_dim: int = -2,
    channel_dim: int | tuple[int, ...] = -1,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
) -> Tensor:  # Float[...]
    r"""Mean euclidean error.

    .. math:: med(x̂，x) ≔ \tfrac{1}{T} ∑ₜ ‖x̂ₜ - x₂‖₂ = ‖x̂ - x‖_{2,1⁎}

    the masked version is defined as

    .. math:: med(x̂，x) ≔ \tfrac{1}{∑ₛ⋁ₖmₛₖ} ∑ₜ‖mₜ⊙(x̂ₜ - xₜ)‖₂

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

    .. math:: 𝗋𝗆𝗌𝖾(x̂，x) ≔ \sqrt{\tfrac{1}{T} ∑ₜ ‖x̂ₜ - x₂‖₂²} = ‖x̂ - x‖_{2,2⁎}

    the masked version is defined as

    .. math:: 𝗋𝗆𝗌𝖾(x̂，x) ≔ \sqrt{\tfrac{1}{∑ₜ⋁ₖmₜₖ} ∑ₜ ‖mₜ ⊙(x̂ₜ - xₜ)‖₂² }

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

    .. math:: 𝗆𝗌𝖾(x̂，x) ≔ \tfrac{1}{T} ∑ₜ ‖x̂ₜ - xₜ‖₂² = ‖x̂ - x‖²_{2,2⁎}

    the masked version is defined as

    .. math:: 𝗆𝗌𝖾(x̂，x) ≔ \tfrac{1}{∑ₜ⋁ₖmₜₖ} ∑ₜ ‖mₜ⊙(x̂ₜ - xₜ)‖₂²

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

    .. math:: nd(x̂，x) ≔ \frac{ ∑ₜ‖x̂ₜ - xₜ‖₁ }{ ∑ₜ‖xₜ‖₁ } = \frac{‖x̂-x‖_{1,1⁎}}{‖x‖_{1,1⁎}}

    the masked version is defined as

    .. math:: nd(x̂，x) ≔ \frac{ ∑ₜ‖mₜ⊙(x̂ₜ - xₜ)‖₁ }{ ∑ₜ‖mₜ⊙xₜ‖₁ }

    where $m ⊙ z ≔ \where(m, z, 0)$ denotes elementwise masked selection. Thus,
    masked ``NaN`` and infinite entries do not contribute; in particular, $0 ⊙ 𝙽𝚊𝙽 = 0$.

    Note:
        This loss is equivalent to `lp_loss` with ``p=1``, ``relative=True``,
        and ``normalization=observation_count``.

        .. math:: nd(x̂，x) ≔ \frac{‖x̂ - x‖_{1⁎}}{‖x‖_{1⁎}}

        where $‖x‖_{p⁎} ≔ \sqrt[p]{(1/T)∑ₜₖ|xₜₖ|ᵖ}$ is a scaled $p$-norm of the tensor.
    """


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

    .. math:: nrmse(x̂，x) ≔ \frac{ \sqrt{ T⁻¹ ∑ₜ‖x̂ₜ - xₜ‖² } }{ T⁻¹ ∑ₜ‖xₜ‖₁ }

    the masked version is defined as

    .. math:: nrmse(x̂，x) ≔ \frac
        { \sqrt{ \tfrac{1}{∑ₜₖmₜₖ} ∑ₜ‖mₜ⊙(x̂ₜ - xₜ)‖₂² } }
        { \tfrac{1}{∑ₜₖmₜₖ} ∑ₜ‖mₜ⊙xₜ‖₁ }

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

    which is equivalent to `lp_loss` with ``p=2`` and ``relative=1``.

    .. math:: 𝖭𝖱𝖬𝖲𝖤(x̂，x) ≔ \frac{ ‖x̂-x‖_{2⁎} }{ ‖x‖_{1⁎} }

    where $‖x‖_{p⁎} ≔ \sqrt[p]{ (1/T)∑ₜₖ|xₜₖ|ᵖ }$ is a scaled $p$-norm of the tensor.

    """


class MAE_Loss(BaseSequenceLoss):
    r"""Mean absolute error.

    .. math:: mae(x̂，x) ≔ \tfrac{1}{T} ∑ₜ ‖x̂ₜ - x₂‖₁

    the masked version is defined as

    .. math:: mae(x̂，x) ≔ \tfrac{1}{∑ₜ⋁ₖmₜₖ} ∑ₜ ‖mₜ⊙(x̂ₜ - xₜ)‖₁

    where $m ⊙ z ≔ \where(m, z, 0)$ denotes elementwise masked selection. Thus,
    masked ``NaN`` and infinite entries do not contribute; in particular, $0 ⊙ 𝙽𝚊𝙽 = 0$.
    """

    reduction: Reduction

    def __init__(
        self, *, time_dim, channel_dim, reduction: Reduction = Reduction.MEAN
    ) -> None:
        super().__init__(time_dim=time_dim, channel_dim=channel_dim)
        self.reduction = reduction

    def forward(
        self,
        *,
        predictions: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
        sample_weights: Tensor | None = None,
    ) -> Tensor:
        errors = mae(
            predictions=predictions,
            targets=targets,
            mask=mask,
            time_dim=self.time_dim,
            channel_dim=self.channel_dim,
        )
        return self.reduction.apply(errors, weights=sample_weights)


class RMSE_Loss(BaseSequenceLoss):
    r"""Root mean square error.

    .. math:: 𝗋𝗆𝗌𝖾(x̂，x) ≔ 𝔼[ \sqrt{\tfrac{1}{T} ∑ₜ ‖x̂ₜ - x₂‖₂²} ]

    the masked version is defined as

    .. math:: 𝗋𝗆𝗌𝖾(x̂，x) ≔ 𝔼[ \sqrt{\tfrac{1}{∑ₜ⋁ₖmₜₖ} ∑ₜ ‖mₜ ⊙(x̂ₜ - xₜ)‖₂² } ]

    where $m ⊙ z ≔ \where(m, z, 0)$ denotes elementwise masked selection. Thus,
    masked ``NaN`` and infinite entries do not contribute; in particular, $0 ⊙ 𝙽𝚊𝙽 = 0$.

    Args:
        time_dim: Tensor axis containing the time steps.
        channel_dim: Tensor axes containing the channels; use ``()`` for a
            univariate time series.
        reduction:
    """

    reduction: Reduction

    def __init__(
        self, *, time_dim, channel_dim, reduction: Reduction = Reduction.MEAN
    ) -> None:
        super().__init__(time_dim=time_dim, channel_dim=channel_dim)
        self.reduction = reduction

    def forward(
        self,
        *,
        predictions: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
        sample_weights: Tensor | None = None,
    ) -> Tensor:
        errors = rmse(
            predictions=predictions,
            targets=targets,
            mask=mask,
            time_dim=self.time_dim,
            channel_dim=self.channel_dim,
        )
        return self.reduction.apply(errors, weights=sample_weights)


def quantile_loss(
    *,
    predictions: Tensor,  # Float[..., $N, 1]
    targets: Tensor,  # Float[..., $N, 1]
    q: float = 0.5,
    mask: Tensor | None = None,
    relative: bool = True,
) -> Tensor:  # Float[()]
    r"""Return the q-quantile loss.

    Given a collection of of univariate time series $x̂ₜ⁽ⁱ⁾$:

    .. math:: 𝖰𝖫_q(x̂，x) ≔ 2\frac{ ∑ᵢₜ ρ_q(x̂ₜ⁽ⁱ⁾，xₜ⁽ⁱ⁾) }{ ∑ᵢₜ|xₜ⁽ⁱ⁾| }

    In order to prevent accidental application on multivariate time series,
    this method checks for the presence of a trailing singleton dimension.

    If a mask is given, instead compute:

    .. math:: 𝖰𝖫_q(x̂，x) ≔
        2\frac{ ∑ᵢₜ ⟦ mₜ⁽ⁱ⁾ \? ρ_q(x̂ₜ⁽ⁱ⁾，xₜ⁽ⁱ⁾) : 0 ⟧ }{ ∑ᵢₜ⟦ mₜ⁽ⁱ⁾ \? |xₜ⁽ⁱ⁾| : 0 ⟧ }

    Args:
        predictions: Predicted values of shape ``[..., N, 1]``.
        targets: Target values of shape ``[..., N, 1]``.
        q: Quantile level in the closed interval ``[0, 1]``.
        mask: Optional Boolean mask of observations to include. It must be
            broadcastable to ``predictions`` and may include the trailing
            singleton channel dimension.
        relative: Whether to normalize the summed quantile errors by the sum
            of absolute target values.

    References:
        - | Deep State Space Models for Time Series Forecasting
          | Syama Sundar Rangapuram et al.
          | Advances in Neural Information Processing Systems 31 (NeurIPS 2018)
          | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """
    if targets.shape[-1] != 1 or predictions.shape[-1] != 1:
        raise ValueError("This loss is only defined for univariate time series.")

    predictions = predictions.squeeze(-1)
    targets = targets.squeeze(-1)

    if mask is not None:
        mask = mask.squeeze(-1)
        predictions = torch.where(mask, predictions, 0.0)
        targets = torch.where(mask, targets, 0.0)

    q_err = 2 * quantile_error(predictions - targets, q=q)
    return q_err.sum() / targets.abs().sum() if relative else q_err.sum()


class Quantile_Loss(BaseSequenceLoss):
    r"""The q-quantile loss.

    .. math:: 𝖰𝖫_q(x̂，x) ≔ 2\frac{ ∑ₜₖρ_q(x̂ₜₖ，xₜₖ) }{ ∑ₜₖ|xₜₖ| }

    References:
        - | Deep State Space Models for Time Series Forecasting
          | Syama Sundar Rangapuram et al.
          | Advances in Neural Information Processing Systems 31 (NeurIPS 2018)
          | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """

    def forward(
        self, *, predictions: Tensor, targets: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        r"""Compute the loss value."""
        return quantile_loss(
            predictions=predictions,
            targets=targets,
            mask=mask,
        )


class LP_Loss(BaseSequenceLoss):
    r"""LP-loss for sequence spaces."""


class SequentialMSE(BaseSequenceLoss):
    r"""Time-Series Mean Square Error.

    Given two random sequences $x,x̂∈( ℝ ∪ \{𝙽𝙰\})^{T×K}$, the time-series
    mean square error is defined as:

    .. math:: 𝖳𝖲-𝖬𝖲𝖤(x̂，x) ≔ ∑ₜₖ \frac{[mₜₖ \? (x̂ₜₖ - xₜₖ)² : 0]}{∑ₛ mₛₖ}

    Or, more precisely, to avoid division by zero, we use the following

    .. math:: ∑ₜₖ \Bigr[ ∑ₛmₛₖ > 0 \? \frac{[mₜₖ \? (x̂ₜₖ - xₜₖ)² : 0]}{∑ₛmₛₖ} : 0\Bigl]

    By default, each channel is normalized by the number of observations in that channel.
    Other normalization schemes are possible, e.g. by the number of observations in the
    entire time series, or by the number of observations in each time step:

    With time-normalization:

    .. math:: ∑ₜₖ \frac{[mₜₖ \? |x̂ₜₖ - xₜₖ|² : 0]}{∑ₛ mₛₖ}

    with channel-normalization:

    .. math:: ∑ₜₖ \frac{[mₜₖ \? |x̂ₜₖ - xₜₖ|² : 0]}{∑ⱼ mₜⱼ}

    with both:

    .. math:: ∑ₜₖ \frac{[mₜₖ \? |x̂ₜₖ - xₜₖ|² : 0]}{∑ₛⱼ mₛⱼ}

    Moreover, we can consider adding a discount factor with respect to the time,
    i.e. a simple geometric dsitribution, which amounts to adding a term of the form
    $γ^{∑ₖ ∆tₖ}$ to the denominator, where $γ$ is the discount factor and $∆t_k$
    is the time difference between the $k$-th and $(k+1)$-th time point.

    Possible batch-dimensions are averaged over.
    """

    # Float[..., $N, *D], Float[..., $N, *D] -> Float[()]
    def forward(
        self, *, predictions: Tensor, targets: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        w = self.channel_weights
        m = ~targets.isnan()  # 1 if not nan, 0 if nan
        r = predictions - targets
        r = torch.where(m, r, 0.0)
        # must come after where, else we get NaN gradients!
        r = r**2 if w is None else w * r**2

        # compute normalization constant
        match self.normalize_time, self.normalize_channels:
            case True, True:
                c = torch.sum(
                    m if w is None else w * m, dim=self.combined_dim, keepdim=True
                )
                s = torch.sum(r / c, dim=self.combined_dim, keepdim=True)
                r = torch.where(c > 0, s, 0.0)

            case True, False:
                c = torch.sum(m, dim=self.time_dim, keepdim=True)
                s = torch.sum(r / c, dim=self.time_dim, keepdim=True)
                r = torch.where(c > 0, s, 0.0)
                r = torch.sum(r, dim=self.channel_dim, keepdim=True)

            case False, True:
                c = torch.sum(
                    m if w is None else w * m, dim=self.channel_dim, keepdim=True
                )
                s = torch.sum(r / c, dim=self.channel_dim, keepdim=True)
                r = torch.where(c > 0, s, 0.0)
                r = torch.sum(r, dim=self.time_dim, keepdim=True)

            case False, False:
                r = torch.sum(r, dim=self.combined_dim, keepdim=True)

            case _:
                raise RuntimeError("unreachable")

        # aggregate over batch-dimensions
        r = torch.mean(r)
        return r
