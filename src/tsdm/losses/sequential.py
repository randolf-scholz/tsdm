r"""Loss functions for time series data.

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
    "Normalization",
    # Classes
    "ND",
    "NRMSE",
    "Quantile_Loss",
    "SequentialMSE",
    # "TimeSeriesMAE",
    # "TimeSeriesRMSE",
    "lp_loss",
    "lp_norm",
    "nd",
    "nrmse",
    "quantile_loss",
]

from enum import StrEnum
from math import prod

import torch
from torch import Tensor

from .base import BaseSequenceLoss
from .samplewise import Reduction, quantile_error

type Dim = int | tuple[int, ...] | None


class Normalization(StrEnum):
    r"""Built-in normalization schemes for sequential norms."""

    SEQUENCE_LENGTH = "sequence_length"
    CHANNEL_PREVALENCE = "channel_prevalence"
    TIMEPOINT_COVERAGE = "timepoint_coverage"
    OBSERVATION_COUNT = "observation_count"

    def compute_normalization(
        self,
        values: Tensor,
        /,
        *,
        time_dim: int,
        channel_dim: int | tuple[int, ...],
        mask: Tensor | None = None,
    ) -> Tensor | int:
        match self:
            case Normalization.SEQUENCE_LENGTH:
                return (
                    values.shape[time_dim]
                    if mask is None
                    else (
                        mask.any(dim=channel_dim, keepdim=True)
                        .sum(dim=time_dim, keepdim=True)
                        .clamp_min(1)
                    )
                )

            case Normalization.CHANNEL_PREVALENCE:
                return (
                    values.shape[time_dim]
                    if mask is None
                    else mask.sum(dim=time_dim, keepdim=True).clamp_min(1)
                )

            case Normalization.TIMEPOINT_COVERAGE:
                return (
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
                return (
                    values.shape[time_dim] * prod(values.shape[d] for d in channel_dim)
                    if mask is None
                    else mask.sum(dim=(time_dim, *channel_dim), keepdim=True).clamp_min(
                        1
                    )
                )


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
            normalizer = scheme.compute_normalization(
                values, mask=mask, time_dim=time_dim, channel_dim=ch_dims
            )
            values = values / normalizer

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


def lp_loss(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    time_dim: int,
    channel_dim: int | tuple[int, ...],  # *D
    # optional args
    p: float = 2.0,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
    channel_weight: Tensor | None = None,  # Float[*D]
    time_weight: Tensor | None = None,  # Float[..., $N]
    normalization: Normalization | Tensor | None = Normalization.CHANNEL_PREVALENCE,
    reduction: Reduction = Reduction.MEAN,
    relative: bool = False,
) -> Tensor:  # Float[()]
    r"""Compute the reduced time-normalized $p$-norm of prediction residuals.

    .. math:: ℓₚ(x̂, x) ≔ 𝔼[‖x̂ - x‖ₚ]

    The arguments other than ``predictions``, ``targets``, and ``reduction``
    are forwarded to :func:`lp_norm`. ``reduction`` aggregates over the batch
    dimensions of the resulting norms.

    Args:
        predictions: Predicted time series.
        targets: Target time series.
        p: Order of the norm.
        mask: Boolean mask indicating valid values.
        channel_dim: Tensor axes of the channel dimensions.
        channel_weight: Optional weights applied within each channel norm.
        time_dim: Tensor axis of the time dimension.
        time_weight: Optional weights applied to time steps.
        normalization: Time-series normalization applied within each norm.
        reduction: Aggregation applied to the per-sequence losses.
        relative: If ``True``, divide each residual norm by the corresponding
            target norm, computing $‖x̂ - x‖ₚ / ‖x‖ₚ$.
    """
    norms = lp_norm(
        predictions - targets,
        p=p,
        mask=mask,
        channel_dim=channel_dim,
        channel_weight=channel_weight,
        time_dim=time_dim,
        time_weight=time_weight,
        normalization=normalization,
    )
    if relative:
        norms = norms / lp_norm(
            targets,
            p=p,
            mask=mask,
            channel_dim=channel_dim,
            channel_weight=channel_weight,
            time_dim=time_dim,
            time_weight=time_weight,
            normalization=normalization,
        )
    match reduction:
        case Reduction.SUM:
            return norms.sum()
        case Reduction.MEAN:
            return norms.mean()
        case Reduction.NONE:
            return norms
        case _:
            raise ValueError(f"Invalid reduction: {reduction}")


def nd(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    time_dim: int = -2,
    channel_dim: Dim = -1,
    # optional args
    mask: Tensor | None = None,  # Bool[..., $N, *D]
    eps: float = 2**-24,
) -> Tensor:  # Float[...]
    r"""Compute the normalized deviation score.

    .. math:: 𝖭𝖣(x̂，x) ≔ \frac{∑ₜₖ|x̂ₜₖ - xₜₖ|}{∑ₜₖ|xₜₖ|}

    or, more generally:

    .. math:: 𝖭𝖣(x̂，x) ≔ \frac{‖x̂ - x‖_{1⁎}}{‖x‖_{1⁎}}

    where $‖x‖_{p⁎} ≔ \sqrt[p]{(1/T)∑ₜₖ|xₜₖ|ᵖ}$ is a scaled $p$-norm of the tensor.

    References:
        - | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
          | Hsiang-Fu Yu, Nikhil Rao, Inderjit S. Dhillon
          | Advances in Neural Information Processing Systems 29 (NIPS 2016)
          | https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html
        - | N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
          | https://openreview.net/forum?id=r1ecqn4YwB
    """
    x_pred = predictions
    x_true = targets
    res = torch.sum((x_pred - x_true).abs(), dim=(-2, -1))
    mag = torch.sum(x_true.abs(), dim=(-2, -1))
    mag = torch.maximum(mag, torch.full_like(mag, eps))
    return torch.mean(res / mag)  # get rid of any batch dimensions


def nrmse(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    time_dim: int = -2,
    channel_dim: Dim = -1,
    # optional args
    mask: Tensor | None = None,  # Bool[..., $N, *D]
    eps: float = 2**-24,
) -> Tensor:  # Float[...]
    r"""Compute the normalized root mean square error as defined by [TRMF]_.

    .. math:: 𝖭𝖱𝖬𝖲𝖤(x̂，x) ≔ \frac{\sqrt{\frac{1}{T}∑ₜₖ|x̂ₜₖ - xₜₖ|²}}{\frac{1}{T}∑ₜₖ|xₜₖ|}

    or, more generally:

    .. math:: 𝖭𝖱𝖬𝖲𝖤(x̂，x) ≔ \frac{ ‖x̂-x‖_{2⁎} }{ ‖x‖_{1⁎} }

    where $‖x‖_{p⁎} ≔ \sqrt[p]{ (1/T)∑ₜₖ|xₜₖ|ᵖ }$ is a scaled $p$-norm of the tensor.

    References:
        .. [TRMF]
        | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
        | Hsiang-Fu Yu, Nikhil Rao, Inderjit S. Dhillon
        | Advances in Neural Information Processing Systems 29 (NIPS 2016)
        | https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html
    """
    x_pred = predictions
    x_true = targets
    res = torch.sqrt(torch.sum(torch.abs(x_pred - x_true) ** 2, dim=(-2, -1)))
    mag = torch.sum(x_true.abs(), dim=(-2, -1))
    mag = torch.maximum(mag, torch.full_like(x_true, eps))

    return torch.mean(res / mag)  # get rid of any batch dimensions


def quantile_loss(
    *,
    predictions: Tensor,
    targets: Tensor,
    q: float = 0.5,
    mask: Tensor | None = None,
) -> Tensor:
    r"""Return the q-quantile loss.

    .. math:: 𝖰𝖫_q(x̂，x) ≔ 2\frac{ ∑ₜₖ ρ_q(x̂ₜₖ，xₜₖ) }{ ∑ₜₖ|xₜₖ| }

    References:
        - | Deep State Space Models for Time Series Forecasting
          | Syama Sundar Rangapuram et al.
          | Advances in Neural Information Processing Systems 31 (NeurIPS 2018)
          | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """
    return (
        2
        * torch.sum(quantile_error(predictions=predictions, targets=targets, q=q))
        / torch.sum(targets.abs())
    )


class ND(BaseSequenceLoss):
    r"""Compute the normalized deviation score.

    .. math:: 𝖭𝖣(x̂，x) ≔ \frac{∑ₜₖ|x̂ₜₖ - xₜₖ|}{∑ₜₖ|xₜₖ|}

    TODO: How to distinguish batch univariate vs single multivariate?
    => Batch makes little sense since all could have different length!

    References:
        - | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
          | https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html
        - | N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
          | https://openreview.net/forum?id=r1ecqn4YwB
    """

    # Float[..., $N], Float[..., $N] -> Float[()]
    def forward(
        self, *, predictions: Tensor, targets: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        return nd(
            predictions=predictions,
            targets=targets,
            mask=mask,
            time_dim=self.time_dim,
            channel_dim=self.channel_dim,
        )


class NRMSE(BaseSequenceLoss):
    r"""Compute the normalized root mean squared error.

    .. math:: 𝖭𝖱𝖬𝖲𝖤(x̂，x) ≔ \frac{\sqrt{\frac{1}{T}∑ₜₖ|x̂ₜₖ - xₜₖ|²}}{∑ₜₖ|xₜₖ|}

    References:
        - | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
          | https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html
    """

    def forward(
        self, *, predictions: Tensor, targets: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        r"""Compute the loss value."""
        return nrmse(
            predictions=predictions,
            targets=targets,
            mask=mask,
            time_dim=self.time_dim,
            channel_dim=self.channel_dim,
        )


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
            time_dim=self.time_dim,
            channel_dim=self.channel_dim,
        )


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
