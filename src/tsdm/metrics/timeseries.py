r"""Loss functions for time series.

Note:
    Contains losses in modular form.
    See `tsdm.metrics.functional` for functional implementations.
"""

__all__ = [
    # ABCs & Protocols
    "TimeSeriesLoss",
    "TimeSeriesBaseLoss",
    "WeightedTimeSeriesLoss",
    # Classes
    "ND",
    "NRMSE",
    "Q_Quantile",
    "Q_Quantile_Loss",
    "TimeSeriesMSE",
    "TimeSeriesWMSE",
    # "TimeSeriesMAE",
    # "TimeSeriesWMAE",
    # "TimeSeriesRMSE",
    # "TimeSeriesWRMSE",
]

from abc import abstractmethod
from collections.abc import Callable
from typing import Final, Optional, Protocol

import torch
from torch import Tensor, nn

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
    time_axis: Final[tuple[int, ...]]
    r"""CONST: The time-axes over which the loss is computed."""
    channel_axis: Final[tuple[int, ...]]
    r"""CONST: The channel-axes over which the loss is computed."""
    combined_axis: Final[tuple[int, ...]]
    r"""CONST: The combined time- and channel-axes."""
    normalize_time: Final[bool]
    r"""CONST: Whether to normalize the weights."""
    normalize_channels: Final[bool]
    r"""CONST: Whether to normalize the weights."""

    def __init__(
        self,
        /,
        *,
        time_axis: int | tuple[int, ...] = -2,
        axis: int | tuple[int, ...] = -1,
        normalize_time: bool = True,
        normalize: bool = False,
    ) -> None:
        super().__init__(axis=axis, normalize=normalize)
        self.normalize_time = bool(normalize_time)
        self.normalize_channels = bool(normalize)
        self.time_axis = (
            (time_axis,) if isinstance(time_axis, int) else tuple(time_axis)
        )
        self.channel_axis = (axis,) if isinstance(axis, int) else tuple(axis)
        self.combined_axis = self.time_axis + self.channel_axis

        if not set(self.time_axis).isdisjoint(self.channel_axis):
            raise ValueError("Time and channel axes must be disjoint!")

    @abstractmethod
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r"""Compute the loss."""
        raise NotImplementedError


class WeightedTimeSeriesLoss(TimeSeriesBaseLoss):
    r"""Base class for a weighted time series loss function.

    Because the loss is computed over a sequence of variable length, the default is to normalize
    the loss by the sequence length, so that loss values are comparable across sequences.
    This class can be used to express decomposable losses of the form

    .. math:: 𝓛(𝐭，x，x̂) ≔ ∑_t ω_t ℓ(x_t，x̂_t；w)

    Where $w$ are the channel weights, and $ω_t$ is a time-dependent discount factor,
    and $ℓ$ is a time-independent loss function. Typically, these losses take the form:

    .. math:: 𝓛(x，x̂)  = ∑_t ω_t Φ(w ⊙ (x_t - x̂_t))

    where $Φ$ is some function acting on the weighted residuals, for example, $Φ(r) = ‖r‖$.
    """

    # Parameters
    channel_weights: Tensor
    r"""PARAM: The weight-vector."""
    discount_factor: Optional[Tensor] = None
    r"""PARAM: The weight-vector."""
    discount_function: Optional[Callable[[Tensor], Tensor]] = None
    r"""Optional: Use a more complicated discounting schema."""

    # Constants
    learnable: Final[bool]
    r"""CONST: Whether the weights are learnable."""

    def __init__(
        self,
        weight: Tensor,
        /,
        *,
        time_axis: Axis = None,
        axis: Axis = None,
        normalize: bool = False,
        normalize_time: bool = True,
        learnable: bool = False,
    ) -> None:
        r"""Initialize the loss function."""
        w = torch.as_tensor(weight, dtype=torch.float32)
        if not torch.all(w >= 0) and torch.any(w > 0):
            raise ValueError(
                "Weights must be non-negative and at least one must be positive."
            )
        axis = tuple(range(-w.ndim, 0)) if axis is None else axis
        time_axis = (-w.ndim - 1,) if time_axis is None else time_axis
        super().__init__(
            time_axis=time_axis,
            axis=axis,
            normalize=normalize,
            normalize_time=normalize_time,
        )

        # Set the weight tensor.
        self.learnable = bool(learnable)
        self.weight = nn.Parameter(w / torch.sum(w), requires_grad=self.learnable)

        # Validate the axes.
        if len(self.channel_axis) != self.weight.ndim:
            raise ValueError(
                "Number of axes does not match weight shape:"
                f" {len(self.channel_axis)} != {self.weight.ndim=}"
            )

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

    .. math:: 𝖭𝖱𝖬𝖲𝖤(x，x̂) ≔ \frac{\sqrt{\frac{1}{T}∑̂ₜₖ |x̂̂ₜₖ - x̂ₜₖ|² }}{∑̂ₜₖ |x̂ₜₖ|}

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
        r = predictions - targets

        m = ~torch.isnan(targets)  # 1 if not nan, 0 if nan
        r = torch.where(m, r, 0.0)
        r = r**2  # must come after where, else we get NaN gradients!

        # compute normalization constant
        # NOTE: JIT does not support match-case
        if self.normalize_time and self.normalize_channels:
            c = torch.sum(m, dim=self.combined_axis, keepdim=True)
            s = torch.sum(r / c, dim=self.combined_axis, keepdim=True)
            r = torch.where(c > 0, s, 0.0)
        elif self.normalize_time and not self.normalize_channels:
            c = torch.sum(m, dim=self.time_axis, keepdim=True)
            s = torch.sum(r / c, dim=self.time_axis, keepdim=True)
            r = torch.where(c > 0, s, 0.0)
            r = torch.sum(r, dim=self.channel_axis, keepdim=True)
        elif not self.normalize_time and self.normalize_channels:
            c = torch.sum(m, dim=self.channel_axis, keepdim=True)
            s = torch.sum(r / c, dim=self.channel_axis, keepdim=True)
            r = torch.where(c > 0, s, 0.0)
            r = torch.sum(r, dim=self.time_axis, keepdim=True)
        else:
            r = torch.sum(r, dim=self.combined_axis, keepdim=True)

        # aggregate over batch-dimensions
        r = torch.mean(r)
        return r


class TimeSeriesWMSE(WeightedTimeSeriesLoss):
    r"""Weighted Time-Series Mean Square Error.

    Given two random sequences $x,x̂∈ℝ^{T×K}$, the weighted time-series mean square error is defined as:

    .. math:: 𝗐𝖳𝖲-𝖬𝖲𝖤(x，x̂) ≔ ∑̂ₜₖ \frac{[m̂ₜₖ \? w_k (x̂̂ₜₖ - x̂ₜₖ)² : 0]}{∑_τ m_{τk}}

    Or, more precisely, to avoid division by zero, we use the following

    .. math:: ∑̂ₜₖ\Bigl[∑_τ m_{τk}>0 \? \frac{[m̂ₜₖ \? w_k (x̂̂ₜₖ - x̂ₜₖ)² : 0]}{∑_τ m_{τk}}:0\Bigr]

    Possible batch-dimensions are averaged over.
    """

    @torch.compile(fullgraph=True)
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r""".. signature:: ``[(..., t, m), (..., t, m)] → ...``."""
        r = predictions - targets

        m = ~torch.isnan(targets)  # 1 if not nan, 0 if nan
        r = torch.where(m, r, 0.0)
        r = self.weight * r**2  # must come after where, else we get NaN gradients!

        # compute normalization constant
        if self.normalize_time and self.normalize_channels:
            c = torch.sum(
                self.weight * m, dim=self.time_axis + self.channel_axis, keepdim=True
            )
            s = torch.sum(r / c, dim=self.time_axis + self.channel_axis, keepdim=True)
            r = torch.where(c > 0, s, 0.0)
        elif self.normalize_time and not self.normalize_channels:
            c = torch.sum(m, dim=self.time_axis, keepdim=True)
            s = torch.sum(r / c, dim=self.time_axis, keepdim=True)
            r = torch.where(c > 0, s, 0.0)
            r = torch.sum(r, dim=self.channel_axis, keepdim=True)
        elif not self.normalize_time and self.normalize_channels:
            c = torch.sum(self.weight * m, dim=self.channel_axis, keepdim=True)
            s = torch.sum(r / c, dim=self.channel_axis, keepdim=True)
            r = torch.where(c > 0, s, 0.0)
            r = torch.sum(r, dim=self.time_axis, keepdim=True)
        else:
            r = torch.sum(r, dim=self.time_axis + self.channel_axis, keepdim=True)

        # aggregate over batch-dimensions
        r = torch.mean(r)
        return r
