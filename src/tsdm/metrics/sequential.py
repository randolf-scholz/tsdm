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
    # ABCs & Protocols
    # Classes
    "ND",
    "NRMSE",
    "Q_Quantile_Loss",
    "SequentialMSE",
    # "TimeSeriesMAE",
    # "TimeSeriesRMSE",
    "Normalization",
    "lp_norm",
    "lp_loss",
    "nd",
    "nrmse",
    "q_quantile_loss",
]

from enum import StrEnum
from math import prod

import torch
from torch import Tensor

from .base import SequentialBaseMetric
from .samplewise import q_quantile

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
        mask: Tensor | None,
        time_dim: int,
        channel_dims: tuple[int, ...],
    ) -> Tensor | int:
        match self:
            case Normalization.SEQUENCE_LENGTH:
                return (
                    values.shape[time_dim]
                    if mask is None
                    else (
                        mask.any(dim=channel_dims, keepdim=True)
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
                    values.shape[time_dim] * prod(values.shape[d] for d in channel_dims)
                    if mask is None
                    else (
                        mask.any(dim=channel_dims, keepdim=True)
                        .sum(dim=time_dim, keepdim=True)
                        .mul(mask.sum(dim=channel_dims, keepdim=True))
                        .clamp_min(1)
                    )
                )

            case Normalization.OBSERVATION_COUNT:
                return (
                    values.shape[time_dim] * prod(values.shape[d] for d in channel_dims)
                    if mask is None
                    else mask.sum(
                        dim=(time_dim, *channel_dims), keepdim=True
                    ).clamp_min(1)
                )


def lp_norm(
    x: Tensor,  # Float[..., $N, *D]
    /,
    *,
    p: float = 2.0,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
    channel_dim: int | tuple[int, ...],  # *D
    channel_weight: Tensor | None = None,  # Float[*D]
    time_dim: int,
    time_weight: Tensor | None = None,  # Float[..., $N]
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
                values, mask=mask, time_dim=time_dim, channel_dims=ch_dims
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
    x: Tensor,  # Float[..., $N, *D]
    /,
    *,
    p: float = 2.0,
    time_dim: int = -2,
    channel_dim: Dim = -1,
    scale_time: bool = True,
    scale_channels: bool = False,
    mask: Tensor | None = None,  # Bool[..., $N, *D] or Bool[..., $N]
    weight: Tensor | None = None,  # Float[...], non-negative
    channel_weight: Tensor | None = None,  # Float[..., *D], non-negative
    time_weight: Tensor | None = None,  # Float[..., $N], non-negative
) -> Tensor:
    r"""Compute time-normalized lp-norm.

    .. math:: (1/B) ∑ₙ ( ∑ₜ∑ₖ wₖsₙₖ ⟦ mₙₜₖ ? |xₙₜₖ|ᵖ : 0⟧ )^{1/p}

    here, the scaling factors $sₙₖ$ should estimate the prevalence of the $k$-th channel.

    - $sₙₖ = 1$ if ``scale_channel = False``
    - $sₙₖ = (∑ₛmₙₜₖ)⁻¹$ if ``scale_channel = True``
    - $sₙₖ = cₖ$ if a tensor is given. in this case, it is recommended to choose

        .. math:: cₖ ∝ 𝐄_{m∼Dataset}[∑ₜmₜₖ]⁻¹

    Args:
        time_dim: Dimension of the time.
        channel_dim: Dimension of the channels.
        channel_weight: Importance scalar $wₖᶜʰ$ for each channel.
        scale_time: Whether to sum or mean aggregation for the time dimension.
        scale_channels: Whether to sum or mean aggregation for the channel dimension.
    """
    raise NotImplementedError


def nd(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    time_dim: int = -2,
    dim: Dim = -1,
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
    dim: Dim = -1,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
    eps: float = 2**-24,
) -> Tensor:  # Float[...]
    r"""Compute the normalized root mean square errors.

    .. math:: 𝖭𝖱𝖬𝖲𝖤(x̂，x) ≔ \frac{\sqrt{\frac{1}{T}∑ₜₖ|x̂ₜₖ - xₜₖ|²}}{\frac{1}{T}∑ₜₖ|xₜₖ|}

    or, more generally:

    .. math:: 𝖭𝖱𝖬𝖲𝖤(x̂，x) ≔ \frac{‖x̂-x‖_{2⁎}}}{‖x‖_{1⁎}}

    where $‖x‖_{p⁎} ≔ \sqrt[p]{(1/T)∑ₜₖ|xₜₖ|ᵖ}$ is a scaled $p$-norm of the tensor.

    References:
        - | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
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


def q_quantile_loss(
    *,
    predictions: Tensor,
    targets: Tensor,
    q: float = 0.5,
) -> Tensor:
    r"""Return the q-quantile loss.

    .. math:: 𝖰𝖫_q(x̂，x) ≔ 2\frac{∑ₜₖ𝖯_q(x̂ₜₖ，xₜₖ)}{∑ₜₖ|xₜₖ|}

    References:
        - | Deep State Space Models for Time Series Forecasting
          | Syama Sundar Rangapuram, Matthias W. Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang,
            Tim Januschowski
          | Advances in Neural Information Processing Systems 31 (NeurIPS 2018)
          | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """
    return (
        2
        * torch.sum(q_quantile(predictions=predictions, targets=targets, q=q))
        / torch.sum(targets.abs())
    )


class ND(SequentialBaseMetric):
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
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        return nd(predictions=predictions, targets=targets)


class NRMSE(SequentialBaseMetric):
    r"""Compute the normalized root mean squared error.

    .. math:: 𝖭𝖱𝖬𝖲𝖤(x̂，x) ≔ \frac{\sqrt{\frac{1}{T}∑ₜₖ|x̂ₜₖ - xₜₖ|²}}{∑ₜₖ|xₜₖ|}

    References:
        - | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
          | https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html
    """

    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r"""Compute the loss value."""
        return nrmse(predictions=predictions, targets=targets)


class Q_Quantile_Loss(SequentialBaseMetric):
    r"""The q-quantile loss.

    .. math:: 𝖰𝖫_q(x̂，x) ≔ 2\frac{ ∑ₜₖ𝖯_q(x̂ₜₖ，xₜₖ) }{∑ₜₖ|xₜₖ|}

    References:
        - | Deep State Space Models for Time Series Forecasting
          | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """

    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r"""Compute the loss value."""
        return q_quantile_loss(predictions=predictions, targets=targets)


class SequentialMSE(SequentialBaseMetric):
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
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
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
