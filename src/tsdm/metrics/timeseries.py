r"""Loss functions for time series.

Note:
    Contains losses in modular form.
    See `tsdm.metrics.functional` for functional implementations.
"""

__all__ = [
    # ABCs & Protocols
    "TimeSeriesLoss",
    "TimeSeriesBaseLoss",
    # Classes
    "ND",
    "NRMSE",
    "Q_Quantile",
    "Q_Quantile_Loss",
    "TimeSeriesMSE",
    # "TimeSeriesMAE",
    # "TimeSeriesRMSE",
]

from abc import abstractmethod
from typing import Final, Protocol

import torch
from torch import Tensor

from tsdm.types.aliases import Axis

from .base import BaseMetric
from .functional import nd, nrmse, q_quantile, q_quantile_loss


class TimeSeriesLoss(Protocol):
    r"""Protocol for a loss function."""

    def __call__(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r"""Compute a loss between the predictions and the targets.

        .. signature:: ``[(..., *t, 𝐧), (..., *t, 𝐧)] -> 0``

        A time series loss function acts on sequences of variable length.
        Given a collection of pairs of sequences $(x_n,x̂_n)∈⋃_{T∈ℕ}(V⊕V)^T$,
        returns a single scalar. Each pair $(x_n,x̂_n)$ is of equal length $T_n$,
        but different pairs may have different lengths.

        In principle, this means that nested/ragged tensors are required.
        However, for the sake of simplicity, we assume that the tensors are
        padded with missing values, such that they are of equal length.
        """
        ...


class TimeSeriesBaseLoss(BaseMetric):
    r"""Base class for a time-series function.

    Because the loss is computed over a sequence of variable length, the default is to normalize
    the loss by the sequence length, so that loss values are comparable across sequences.
    This class can be used to express decomposable losses of the form

    .. math:: 𝓛(x，x̂) ≔ 𝐀_t ℓ(x_t，x̂_t)

    By default, the aggregation $𝐀_t$ is the mean over the time-axes $𝐄_t$, but simply
    summing over the time-axes is also possible.
    """

    # Constants
    time_axis: Final[int]
    r"""CONST: The time-axes over which the loss is computed."""
    channel_axes: Final[tuple[int, ...]]
    r"""CONST: The channel-axes over which the loss is computed."""
    combined_axes: Final[tuple[int, ...]]
    r"""CONST: The combined time- and channel-axes."""
    normalize_time: Final[bool]
    r"""CONST: Whether to normalize the weights."""
    normalize_channels: Final[bool]
    r"""CONST: Whether to normalize the weights."""

    # TODO: implement discount factors.

    def __init__(
        self,
        /,
        *,
        weight: Tensor | None = None,
        axis: Axis = -1,
        time_axis: int = -2,
        normalize_time: bool = True,
        normalize: bool = False,
        learnable: bool = False,
    ) -> None:
        super().__init__(
            axis=axis,
            normalize=normalize,
            weight=weight,
            learnable=learnable,
        )
        self.channel_axes = self.axis  # alias
        self.normalize_channels = self.normalize  # alias

        self.normalize_time = bool(normalize_time)
        self.time_axis = int(time_axis)
        self.combined_axes = (self.time_axis, *self.channel_axes)

        if not {self.time_axis}.isdisjoint(self.channel_axes):
            raise ValueError("Time and channel axes must be disjoint!")

    @abstractmethod
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r"""Compute the loss."""
        raise NotImplementedError


class ND(TimeSeriesBaseLoss):
    r"""Compute the normalized deviation score.

    .. math:: 𝖭𝖣(x，x̂) ≔ \frac{∑̂ₜₖ |x̂̂ₜₖ - x̂ₜₖ|}{∑̂ₜₖ |x̂ₜₖ|}

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


class NRMSE(TimeSeriesBaseLoss):
    r"""Compute the normalized root mean squared error.

    .. math:: 𝖭𝖱𝖬𝖲𝖤(x，x̂) ≔ \frac{\sqrt{\frac{1}{T}∑̂ₜₖ |x̂̂ₜₖ - x̂ₜₖ|²}}{∑̂ₜₖ |x̂ₜₖ|}

    References:
        - | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
          | https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html
    """

    @torch.compile(fullgraph=True)
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r"""Compute the loss value."""
        return nrmse(predictions=predictions, targets=targets)


class Q_Quantile(TimeSeriesBaseLoss):
    r"""The q-quantile.

    .. math:: 𝖯_q(x，x̂) ≔ \begin{cases}\hfill q⋅|x-x̂|:& x≥x̂ \\ (1-q)⋅|x-x̂|:& x≤x̂ \end{cases}

    References:
        - | Deep State Space Models for Time Series Forecasting
          | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """

    @torch.compile(fullgraph=True)
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r"""Compute the loss value."""
        return q_quantile(predictions=predictions, targets=targets)


class Q_Quantile_Loss(TimeSeriesBaseLoss):
    r"""The q-quantile loss.

    .. math:: 𝖰𝖫_q(x，x̂) ≔ 2\frac{∑̂ₜₖ𝖯_q(x̂ₜₖ，x̂̂ₜₖ)}{∑̂ₜₖ|x̂ₜₖ|}

    References:
        - | Deep State Space Models for Time Series Forecasting
          | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """

    @torch.compile(fullgraph=True)
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r"""Compute the loss value."""
        return q_quantile_loss(predictions=predictions, targets=targets)


class TimeSeriesMSE(TimeSeriesBaseLoss):
    r"""Time-Series Mean Square Error.

    Given two random sequences $x,x̂∈(ℝ∪𝙽𝚊𝙽)^{T×K}$, the time-series mean square error is defined as:

    .. math:: 𝖳𝖲-𝖬𝖲𝖤(x，x̂) ≔ ∑̂ₜₖ \frac{[m̂ₜₖ \? (x̂̂ₜₖ - x̂ₜₖ)² : 0]}{∑_τ m_{τk}}

    Or, more precisely, to avoid division by zero, we use the following

    .. math:: ∑̂ₜₖ \Bigr[∑_τ m_{τk}>0 \? \frac{[m̂ₜₖ \? (x̂̂ₜₖ - x̂ₜₖ)² : 0]}{∑_τ m_{τk}} : 0\Bigl]

    By default, each channel is normalized by the number of observations in that channel.
    Other normalization schemes are possible, e.g. by the number of observations in the
    entire time series, or by the number of observations in each time step:

    With time-normalization:

    .. math:: ∑̂ₜₖ \frac{[m̂ₜₖ \? (x̂̂ₜₖ - x̂ₜₖ)² : 0]}{∑_τ m_{τk}}

    with channel-normalization:

    .. math:: ∑̂ₜₖ \frac{[m̂ₜₖ \? (x̂̂ₜₖ - x̂ₜₖ)² : 0]}{∑_j m_{tj}}

    with both:

    .. math:: ∑̂ₜₖ \frac{[m̂ₜₖ \? (x̂̂ₜₖ - x̂ₜₖ)² : 0]}{∑_{τj} m_{τj}}

    Moreover, we can consider adding a discount factor with respect to the time,
    i.e. a simple geometric dsitribution, which amounts to adding a term of the form
    $γ^{∑_k ∆t_k}$ to the denominator, where $γ$ is the discount factor and $∆t_k$
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
