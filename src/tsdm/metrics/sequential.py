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
    "nd",
    "nrmse",
    "q_quantile_loss",
]

import torch
from torch import Tensor

from . import samplewise
from .base import SequentialBaseMetric
from .samplewise import q_quantile

type Dim = int | tuple[int, ...] | None


def lp_norm(
    x: Tensor,  # Float[..., $N, *D]
    /,
    *,
    p: float = 2.0,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
    channel_dim: Dim = -1,  # *D
    channel_weight: Tensor | None = None,  # Float[*D]
    time_dim: int = -2,
    time_weight: Tensor | None = None,  # Float[..., $N]
    scale_time: bool = True,
    scale_channels: bool = False,
) -> Tensor:  # Float[...]
    r"""Compute time-normalized lp-norm.

    .. math:: ℓₚ(x) = ( ∑ₜ wₜᵗⁱᵐᵉ ∑ₖ wₖᶜʰsₖ ⟦ mₜₖ ? |xₜₖ|ᵖ : 0⟧ )^{1/p}

    here, the scaling factors $sₙₖ$ should estimate the prevalence of the $k$-th channel.

    - $sₖ = 1$ if ``scale_channel = False``
    - $sₖ = (∑ₛmₜₖ)⁻¹$ if ``scale_channel = True``
    - $sₖ = cₖ$ if a tensor is given. in this case, it is recommended to choose

        .. math:: cₖ ∝ 𝐄_{m∼Dataset}[∑ₜmₜₖ]⁻¹

    Args:
        x: Tensor of shape $(\$N, *D)$.
        p: Order of the norm (any non-NAN float including ±inf).
        mask: Boolean mask indicating valid values.
        time_dim: Tensor axis of the time dimension.
        time_weight: Importance weight $wₜᵗⁱᵐᵉ$ for each time step.
        channel_dim: Tensor axes of the channel dimensions.
        channel_weight: Importance weight $wₖᶜʰ$ for each channel.
        scale_time: Whether to sum or mean aggregation for the time dimension.
        scale_channels: Whether to sum or mean aggregation for the channel dimension.
    """
    assert x.ndim >= 1, "x must have at least one dimension"


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
