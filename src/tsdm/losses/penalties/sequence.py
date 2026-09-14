__all__ = ["Normalization", "lp_norm"]

from enum import StrEnum
from math import prod
from typing import assert_never

import torch
from torch import Tensor


class Normalization(StrEnum):
    r"""Built-in normalization schemes for sequential norms."""

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


def lp_norm(
    x: Tensor,  # Float[..., $N, *D]
    /,
    *,
    time_dim: int,
    channel_dim: int | tuple[int, ...],  # *D
    # optional args
    p: float = 2.0,
    channel_weight: Tensor | None = None,  # Float[*D]
    time_weight: Tensor | None = None,  # Float[..., $N]
    mask: Tensor | None = None,  # Bool[..., $N, *D]
    normalization: Normalization | Tensor | None = Normalization.CHANNEL_PREVALENCE,
) -> Tensor:  # Float[...]
    r"""Compute time-normalized lp-norm.

    .. math:: ℓₚ(x) = ( ∑ₜ wₜᵗⁱᵐᵉ ∑ₖ wₖᶜʰsₜₖ⁻¹ ⟦ mₜₖ ? |xₜₖ|ᵖ : 0⟧ )^{1/p}

    Args:
        x: Tensor of shape ``[..., $N, *D]``.
        p: Order of the norm. Note: $p∈\{0, ±∞\}$ are currently not implemented.
        mask: Boolean mask indicating valid values.
        time_dim: Required tensor axis of the time dimension.
        time_weight: Importance weight $wₜᵗⁱᵐᵉ$ for each time step.
        channel_dim: Required tensor axes of the channel dimensions. Use ``()``
            for an univariate time series.
        channel_weight: Importance weight $wₖᶜʰ$ for each channel.
        normalization: Normalization applied to the powered values. ``None``
            performs no normalization ($sₜₖ=1$). The built-in schemes are:

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
    ch_dims = (channel_dim,) if isinstance(channel_dim, int) else channel_dim
    if not all(-x.ndim <= d < x.ndim for d in (time_dim, *ch_dims)):
        raise ValueError(f"channel or time dims are out of range for {x.ndim=}.")

    # normalize
    time_dim = time_dim % x.ndim
    ch_dims = tuple(d % x.ndim for d in ch_dims)

    if x.ndim < 1:
        raise ValueError("x must have at least a time dimension.")
    if time_dim in ch_dims:
        raise ValueError("time_dim and channel_dim must be disjoint.")
    if len(set(ch_dims)) != len(ch_dims):
        raise ValueError("channel_dim must not contain duplicate dimensions.")
    if ch_dims != tuple(sorted(ch_dims)):
        raise ValueError("channel_dim must be strictly increasing.")

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

    if (w_t := time_weight) is not None:
        time_axes = tuple(axis for axis in range(x.ndim) if axis not in ch_dims)
        if not 1 <= w_t.ndim <= len(time_axes):
            raise ValueError(f"Expected 1 <= {w_t.ndim=} <= {len(time_axes)=}")
        dims = time_axes[-w_t.ndim :]
        w_t = w_t[*(slice(None) if d in dims else None for d in range(x.ndim))]
        values = w_t * values

    if (w_k := channel_weight) is not None:
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
