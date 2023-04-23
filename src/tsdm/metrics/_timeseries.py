"""TODO: add package summary.

TODO: add package description.
"""

__all__ = []

from abc import ABCMeta, abstractmethod
from typing import Callable, Final, Optional, Protocol, runtime_checkable

import torch
from torch import Tensor, jit, nn

from tsdm.metrics._modular import WeightedLoss
from tsdm.metrics.functional import nd, nrmse, q_quantile, q_quantile_loss
from tsdm.utils.decorators import autojit


@runtime_checkable
class TimeSeries(Protocol):
    r"""Protocol for a loss function."""

    def __call__(self, targets: Tensor, predictions: Tensor, /) -> Tensor:
        r"""Compute a loss between the targets and the predictions.

        .. Signature:: ``[(..., *t, 𝐧), (..., *t, 𝐧)] -> 0``

        A time series loss function acts on sequences of variable length.
        Given a collection of pairs of sequences $(x_n,x̂_n)∈⋃_{T∈ℕ}(V⊕V)^T$,
        returns a single scalar. Each pair $(x_n,x̂_n)$ is of equal length $T_n$,
        but different pairs may have different lengths.

        In principle this means that nested/ragged tensors are required.
        However, for the sake of simplicity, we assume that the tensors are
        padded with missing values, such that they are of equal length.
        """


class TimeSeriesBaseLoss(nn.Module, metaclass=ABCMeta):
    r"""Base class for a time-series function.

    Because the loss is computed over a sequence of variable length,
    the default is to normalize the loss by the sequence length, so that loss
    values are comparable across sequences.
    """

    # Constants
    axes: Final[tuple[int, ...]]
    r"""CONST: The axes over which the loss is computed."""
    normalize_time: Final[bool]
    r"""CONST: Whether to normalize the weights."""
    normalize_channels: Final[bool]
    r"""CONST: Whether to normalize the weights."""
    rank: Final[int]
    r"""CONST: The number of dimensions over which the loss is computed"""

    def __init__(
        self,
        axes: int | tuple[int, ...] = -1,
        /,
        *,
        normalize_time: bool = True,
        normalize_channels: bool = False,
    ):
        super().__init__()
        self.normalize_time = normalize_time
        self.normalize_channels = normalize_channels
        self.axes = (axes,) if isinstance(axes, int) else tuple(axes)
        self.rank = len(self.axes)

    @abstractmethod
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r"""Compute the loss."""
        raise NotImplementedError


class WeightedTimeSeriesLoss(nn.Module, metaclass=ABCMeta):
    r"""Base class for a weighted time series loss function.

    Because the loss is computed over a sequence of variable length,
    the default is to normalize the loss by the sequence length, so that loss
    values are comparable across sequences.
    """

    # Parameters
    channel_weights: Tensor
    r"""PARAM: The weight-vector."""
    discount_factor: Optional[Tensor] = None
    r"""PARAM: The weight-vector."""
    discount_function: Optional[Callable[[Tensor], Tensor]] = None
    r"""Optional: Use a more complicated discounting schema."""

    # Constants
    axes: Final[tuple[int, ...]]
    r"""CONST: The axes over which the loss is computed."""
    learnable: Final[bool]
    r"""CONST: Whether the weights are learnable."""
    normalize_time: Final[bool]
    r"""CONST: Whether to normalize the weights."""
    normalize_channels: Final[bool]
    r"""CONST: Whether to normalize the weights."""
    rank: Final[int]
    r"""CONST: The number of dimensions of the weight tensor."""
    shape: Final[tuple[int, ...]]
    r"""CONST: The shape of the weight tensor."""

    def __init__(
        self,
        weight: Tensor,
        /,
        *,
        learnable: bool = False,
        normalize: bool = False,
        axes: Optional[tuple[int, ...]] = None,
    ):
        super().__init__()
        self.learnable = learnable
        self.normalize = normalize

        w = torch.as_tensor(weight, dtype=torch.float32)
        assert torch.all(w >= 0) and torch.any(w > 0)
        w = w / torch.sum(w) if self.normalize else w

        self.weight = nn.Parameter(w, requires_grad=self.learnable)
        self.rank = len(self.weight.shape)
        self.shape = tuple(self.weight.shape)
        self.axes = tuple(range(-self.rank, 0)) if axes is None else axes
        assert len(self.axes) == self.rank

    @abstractmethod
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r"""Compute the loss."""
        raise NotImplementedError


@autojit
class ND(TimeSeriesBaseLoss):
    r"""Compute the normalized deviation score.

    .. math:: 𝖭𝖣(x，x̂) ≔ \frac{∑_{tk} |x̂_{tk} - x_{tk}|}{∑_{tk} |x_{tk}|}

    TODO: How to distinguish batch univariate vs single multivariate?
    => Batch makes little sense since all could have different length!

    References:
        - | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
          | https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html
        - | N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
          | https://openreview.net/forum?id=r1ecqn4YwB
    """

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n), (..., n) -> ()``."""
        return nd(targets, predictions)


@autojit
class NRMSE(TimeSeriesBaseLoss):
    r"""Compute the normalized root mean square error.

    .. math:: 𝖭𝖱𝖬𝖲𝖤(x，x̂) ≔ \frac{\sqrt{\frac{1}{T}∑_{tk} |x̂_{tk} - x_{tk}|^2 }}{∑_{tk} |x_{tk}|}

    References:
        - | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
          | https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html
    """

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r"""Compute the loss value."""
        return nrmse(targets, predictions)


@autojit
class Q_Quantile(TimeSeriesBaseLoss):
    r"""The q-quantile.

    .. math:: 𝖯_q(x，x̂) ≔ \begin{cases}\hfill q⋅|x-x̂|:& x≥x̂ \\ (1-q)⋅|x-x̂|:& x≤x̂ \end{cases}

    References:
        - | Deep State Space Models for Time Series Forecasting
          | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r"""Compute the loss value."""
        return q_quantile(targets, predictions)


@autojit
class Q_Quantile_Loss(TimeSeriesBaseLoss):
    r"""The q-quantile loss.

    .. math:: 𝖰𝖫_q(x，x̂) ≔ 2\frac{∑_{tk}𝖯_q(x_{tk}，x̂_{tk})}{∑_{tk}|x_{tk}|}

    References:
        - | Deep State Space Models for Time Series Forecasting
          | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r"""Compute the loss value."""
        return q_quantile_loss(targets, predictions)


@autojit
class TimeSeriesMSE(nn.Module):
    r"""Time-Series Mean Square Error.

    Given two random sequences $x,x̂∈(ℝ∪𝙽𝚊𝙽)^{T×K}$, the time-series mean square error is defined as:

    .. math:: 𝖳𝖲-𝖬𝖲𝖤(x，x̂) ≔ ∑_{tk} \frac{[m_{tk} \? (x̂_{tk} - x_{tk})^2 : 0]}{∑_τ m_{τk}}

    Or, more precisely, to avoid division by zero, we use the following

    .. math:: ∑_{tk} \Bigr[∑_τ m_{τk}>0 \? \frac{[m_{tk} \? (x̂_{tk} - x_{tk})^2 : 0]}{∑_τ m_{τk}} : 0\Bigl]

    By default, each channel is normalized by the number of observations in that channel.
    Other normalization schemes are possible, e.g. by the number of observations in the
    entire time series, or by the number of observations in each time step:

    With time-normalization:

    .. math:: ∑_{tk} \frac{[m_{tk} \? (x̂_{tk} - x_{tk})^2 : 0]}{∑_τ m_{τk}}

    with channel-normalization:

    .. math:: ∑_{tk} \frac{[m_{tk} \? (x̂_{tk} - x_{tk})^2 : 0]}{∑_j m_{tj}}

    with both:

    .. math:: ∑_{tk} \frac{[m_{tk} \? (x̂_{tk} - x_{tk})^2 : 0]}{∑_{τj} m_{τj}}

    Moreover, we can consider adding a discount factor with respect to the time,
    i.e. a simple geometric dsitribution, which amounts to adding a term of the form
    $γ^{∑_k ∆t_k}$ to the denominator, where $γ$ is the discount factor and $∆t_k$
    is the time difference between the $k$-th and $(k+1)$-th time point.

    Possible batch-dimensions are averaged over.
    """

    # Constants
    axes: Final[tuple[int, ...]]
    r"""CONST: The axes over which the loss is computed."""
    discount: Final[float]
    r"""CONST: The discount factor for the time-series."""
    time_axes: Final[tuple[int, ...]]
    r"""CONST: The time axis."""
    normalize_time: Final[bool]
    r"""CONST: Whether to normalize over time."""
    normalize_channels: Final[bool]
    r"""CONST: Whether to normalize the loss by the number of channels."""

    def __init__(
        self,
        axes: int | tuple[int, ...] = -1,
        time_axes: Optional[int | tuple[int, ...]] = None,
        /,
        *,
        discount: float = 1.0,
        normalize_time: bool = True,
        normalize_channels: bool = False,
    ) -> None:
        super().__init__()
        self.axes = (axes,) if isinstance(axes, int) else tuple(axes)
        t_axes = min(self.axes) - 1 if time_axes is None else time_axes
        self.time_axes = (t_axes,) if isinstance(t_axes, int) else tuple(t_axes)
        assert set(self.time_axes).isdisjoint(
            self.axes
        ), "time and channel axes must be disjoint"
        self.discount = discount
        self.normalize_channels = normalize_channels
        self.normalize_time = normalize_time

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., t, 𝐦), (..., t, 𝐦)] → ...``."""
        r = predictions - targets

        m = ~torch.isnan(targets)  # 1 if not nan, 0 if nan
        r = torch.where(m, r, 0.0)
        r = r**2  # must come after where, else we get NaN gradients!

        # compute normalization constant
        if self.normalize_channels and self.normalize_time:
            c = torch.sum(m, dim=self.axes + self.time_axes, keepdim=True)
            s = torch.sum(r / c, dim=self.axes + self.time_axes, keepdim=True)
            r = torch.where(c > 0, s, 0.0)
        elif self.normalize_channels and not self.normalize_time:
            c = torch.sum(m, dim=self.axes, keepdim=True)
            s = torch.sum(r / c, dim=self.axes, keepdim=True)
            r = torch.where(c > 0, s, 0.0)
            r = torch.sum(r, dim=self.time_axes, keepdim=True)
        elif not self.normalize_channels and self.normalize_time:
            c = torch.sum(m, dim=self.time_axes, keepdim=True)
            s = torch.sum(r / c, dim=self.time_axes, keepdim=True)
            r = torch.where(c > 0, s, 0.0)
            r = torch.sum(r, dim=self.axes, keepdim=True)
        else:
            r = torch.sum(r, dim=self.axes + self.time_axes, keepdim=True)

        # aggregate over batch-dimensions
        r = torch.mean(r)
        return r


@autojit
class TimeSeriesWMSE(WeightedLoss):
    r"""Weighted Time-Series Mean Square Error.

    Given two random sequences $x,x̂∈ℝ^{T×K}$, the weighted time-series mean square error is defined as:

    .. math:: 𝗐𝖳𝖲-𝖬𝖲𝖤(x，x̂) ≔ ∑_{tk} \frac{[m_{tk} \? w_k (x̂_{tk} - x_{tk})^2 : 0]}{∑_τ m_{τk}}

    Or, more precisely, to avoid division by zero, we use the following

    .. math:: ∑_{tk}\Bigl[∑_τ m_{τk}>0 \? \frac{[m_{tk} \? w_k (x̂_{tk} - x_{tk})^2 : 0]}{∑_τ m_{τk}}:0\Bigr]

    Possible batch-dimensions are averaged over.
    """

    time_axes: Final[tuple[int, ...]]
    r"""CONST: The time axis."""

    def __init__(
        self,
        weight: Tensor,
        /,
        *,
        learnable: bool = False,
        normalize: bool = False,
        axes: Optional[tuple[int, ...]] = None,
        time_axes: Optional[int | tuple[int, ...]] = None,
    ):
        super().__init__(weight, learnable=learnable, normalize=normalize, axes=axes)
        t_axes = min(self.axes) - 1 if time_axes is None else time_axes
        self.time_axes = (t_axes,) if isinstance(t_axes, int) else tuple(t_axes)

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., t, m), (..., t, m)] → ...``."""
        r = predictions - targets

        m = ~torch.isnan(targets)  # 1 if not nan, 0 if nan
        r = torch.where(m, r, 0.0)
        r = self.weight * r**2  # must come after where, else we get NaN gradients!

        # compute normalization constant
        if self.normalize_channels and self.normalize_time:
            c = torch.sum(self.weight * m, dim=self.axes + self.time_axes, keepdim=True)
            s = torch.sum(r / c, dim=self.axes + self.time_axes, keepdim=True)
            r = torch.where(c > 0, s, 0.0)
        elif self.normalize_channels and not self.normalize_time:
            c = torch.sum(self.weight * m, dim=self.axes, keepdim=True)
            s = torch.sum(r / c, dim=self.axes, keepdim=True)
            r = torch.where(c > 0, s, 0.0)
            r = torch.sum(r, dim=self.time_axes, keepdim=True)
        elif not self.normalize_channels and self.normalize_time:
            c = torch.sum(m, dim=self.time_axes, keepdim=True)
            s = torch.sum(r / c, dim=self.time_axes, keepdim=True)
            r = torch.where(c > 0, s, 0.0)
            r = torch.sum(r, dim=self.axes, keepdim=True)
        else:
            # c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)
            r = torch.sum(r, dim=self.axes + self.time_axes, keepdim=True)

        # # aggregate over time
        # s = torch.sum(r / c, dim=self.axes + self.time_axes, keepdim=True)
        # r = torch.where(c > 0, s, 0.0)

        # aggregate over batch-dimensions
        r = torch.mean(r)
        return r
