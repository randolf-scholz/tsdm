r"""Loss functions for time series.

Note:
    Contains losses in modular form.
    See `tsdm.metrics.functional` for functional implementations.
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

from .base import SequentialBaseMetric
from .samplewise import q_quantile


def nd(*, predictions: Tensor, targets: Tensor, eps: float = 2**-24) -> Tensor:
    r"""Compute the normalized deviation score.

    .. math:: 𝖭𝖣(x̂，x) ≔ \frac{∑ₜₖ|x̂ₜₖ - xₜₖ|}{∑ₜₖ|xₜₖ|}

    TODO: How to distinguish batch univariate vs single multivariate?
    => Batch makes little sense since all could have different length!

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
    mag = torch.maximum(mag, torch.full_like(x_true, eps))
    return torch.mean(res / mag)  # get rid of any batch dimensions


def nrmse(*, predictions: Tensor, targets: Tensor, eps: float = 2**-24) -> Tensor:
    r"""Compute the normalized root mean square errors.

    .. math:: 𝖭𝖱𝖬𝖲𝖤(x̂，x) ≔ \frac{\sqrt{\frac{1}{T}∑ₜₖ|x̂ₜₖ - xₜₖ|²}}{∑ₜₖ|xₜₖ|}

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


def q_quantile_loss(*, predictions: Tensor, targets: Tensor, q: float = 0.5) -> Tensor:
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

    @torch.compile(fullgraph=True)
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r""".. signature:: ``(..., n), (..., n) -> ()``."""
        return nd(predictions=predictions, targets=targets)


class NRMSE(SequentialBaseMetric):
    r"""Compute the normalized root mean squared error.

    .. math:: 𝖭𝖱𝖬𝖲𝖤(x̂，x) ≔ \frac{\sqrt{\frac{1}{T}∑ₜₖ|x̂ₜₖ - xₜₖ|²}}{∑ₜₖ|xₜₖ|}

    References:
        - | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
          | https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html
    """

    @torch.compile(fullgraph=True)
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r"""Compute the loss value."""
        return nrmse(predictions=predictions, targets=targets)


class Q_Quantile_Loss(SequentialBaseMetric):
    r"""The q-quantile loss.

    .. math:: 𝖰𝖫_q(x̂，x) ≔ 2\frac{ ∑ₜₖ𝖯_q(x̂ₜₖ，xₜₖ) }{∑ₜₖ\abs{xₜₖ}}

    References:
        - | Deep State Space Models for Time Series Forecasting
          | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """

    @torch.compile(fullgraph=True)
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r"""Compute the loss value."""
        return q_quantile_loss(predictions=predictions, targets=targets)


class SequentialMSE(SequentialBaseMetric):
    r"""Time-Series Mean Square Error.

    Given two random sequences $x,x̂∈( ℝ ∪ \{𝙽𝙰\})^{T×K}$, the time-series
    mean square error is defined as:

    .. math:: 𝖳𝖲-𝖬𝖲𝖤(x̂，x) ≔ ∑ₜₖ \frac{[mₜₖ \? (x̂ₜₖ - xₜₖ)² : 0]}{∑ₛ mₛₖ}

    Or, more precisely, to avoid division by zero, we use the following

    .. math:: ∑ₜₖ \Bigr[∑ₛ mₛₖ > 0 \? \frac{[mₜₖ \? (x̂ₜₖ - xₜₖ)² : 0]}{∑ₛ mₛₖ} : 0\Bigl]

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

    @torch.compile
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r""".. signature:: ``[(..., t, 𝐦), (..., t, 𝐦)] → ...``."""
        w = self.weight
        m = ~targets.isnan()  # 1 if not nan, 0 if nan
        r = predictions - targets
        r = torch.where(m, r, 0.0)
        # must come after where, else we get NaN gradients!
        r = r**2 if w is None else w * r**2

        # compute normalization constant
        match self.normalize_time, self.normalize_channels:
            case True, True:
                c = torch.sum(
                    m if w is None else w * m, dim=self.combined_axes, keepdim=True
                )
                s = torch.sum(r / c, dim=self.combined_axes, keepdim=True)
                r = torch.where(c > 0, s, 0.0)

            case True, False:
                c = torch.sum(m, dim=self.time_axis, keepdim=True)
                s = torch.sum(r / c, dim=self.time_axis, keepdim=True)
                r = torch.where(c > 0, s, 0.0)
                r = torch.sum(r, dim=self.channel_axes, keepdim=True)

            case False, True:
                c = torch.sum(
                    m if w is None else w * m, dim=self.channel_axes, keepdim=True
                )
                s = torch.sum(r / c, dim=self.channel_axes, keepdim=True)
                r = torch.where(c > 0, s, 0.0)
                r = torch.sum(r, dim=self.time_axis, keepdim=True)

            case False, False:
                r = torch.sum(r, dim=self.combined_axes, keepdim=True)

            case _:
                raise RuntimeError("unreachable")

        # aggregate over batch-dimensions
        r = torch.mean(r)
        return r
