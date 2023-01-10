r"""Implementations of loss functions.

Notes
-----
Contains losses in modular form.

- See `tsdm.losses.functional` for functional implementations.
"""

__all__ = [
    # Base Classes
    "Loss",
    "WeightedLoss",
    # Classes
    "ND",
    "NRMSE",
    "Q_Quantile",
    "Q_Quantile_Loss",
    "WRMSE",
    "RMSE",
    "MSE",
    "WMSE",
    "MAE",
    "WMAE",
    "TimeSeriesMSE",
    "TimeSeriesWMSE",
    # "TimeSeriesMAE",
    # "TimeSeriesWMAE",
    # "TimeSeriesRMSE",
    # "TimeSeriesWRMSE",
]


from abc import ABCMeta, abstractmethod
from typing import Final, Optional

import torch
from torch import Tensor, jit, nn

from tsdm.metrics.functional import nd, nrmse, q_quantile, q_quantile_loss
from tsdm.utils.decorators import autojit


class Loss(nn.Module, metaclass=ABCMeta):
    r"""Base class for a loss function."""

    # Constants
    axes: Final[tuple[int, ...]]
    r"""CONST: The axes over which the loss is computed."""
    normalize: Final[bool]
    r"""CONST: Whether to normalize the weights."""
    rank: Final[int]
    r"""CONST: The number of dimensions over which the loss is computed"""

    def __init__(
        self,
        axes: int | tuple[int, ...] = -1,
        /,
        *,
        normalize: bool = False,
    ):
        super().__init__()
        self.normalize = normalize
        self.axes = (axes,) if isinstance(axes, int) else tuple(axes)
        self.rank = len(self.axes)

    @abstractmethod
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r"""Compute the loss."""
        raise NotImplementedError


class WeightedLoss(nn.Module, metaclass=ABCMeta):
    r"""Base class for a weighted loss function."""

    # Parameters
    weight: Tensor
    r"""PARAM: The weight-vector."""

    # Constants
    axes: Final[tuple[int, ...]]
    r"""CONST: The axes over which the loss is computed."""
    learnable: Final[bool]
    r"""CONST: Whether the weights are learnable."""
    normalize: Final[bool]
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
class ND(Loss):
    r"""Compute the normalized deviation score.

    .. math:: 𝖭𝖣(x, x̂) = \frac{∑_{t,k} |x̂_{t,k} -  x_{t,k}|}{∑_{t,k} |x_{t,k}|}

    TODO: How to distinguish batch univariate vs single multivariate?
    => Batch makes little sense since all could have different length!

    References
    ----------
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
class NRMSE(Loss):
    r"""Compute the normalized root mean square error.

    .. math:: 𝖭𝖱𝖬𝖲𝖤(x, x̂) = \frac{\sqrt{ \frac{1}{T}∑_{t,k} |x̂_{t,k} - x_{t,k}|^2 }}{∑_{t,k} |x_{t,k}|}

    References
    ----------
    - | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
      | https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html
    """

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r"""Compute the loss value."""
        return nrmse(targets, predictions)


@autojit
class Q_Quantile(Loss):
    r"""The q-quantile.

    .. math:: 𝖯_q(x,x̂) = \begin{cases} q |x-x̂|:& x≥x̂ \\ (1-q)|x-x̂|:& x≤x̂ \end{cases}

    References
    ----------
    - | Deep State Space Models for Time Series Forecasting
      | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r"""Compute the loss value."""
        return q_quantile(targets, predictions)


@autojit
class Q_Quantile_Loss(Loss):
    r"""The q-quantile loss.

    .. math:: 𝖰𝖫_q(x,x̂) = 2\frac{∑_{it}𝖯_q(x_{it},x̂_{it})}{∑_{it}|x_{it}|}

    References
    ----------
    - | Deep State Space Models for Time Series Forecasting
      | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r"""Compute the loss value."""
        return q_quantile_loss(targets, predictions)


@autojit
class MAE(Loss):
    r"""Mean Absolute Error.

    .. math:: 𝖬𝖠𝖤(x,x̂) = \sqrt{𝔼[‖x - x̂‖]}
    """

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        r = predictions - targets

        m = ~torch.isnan(targets)
        r = torch.where(m, r, 0.0)
        r = torch.abs(r)
        r = torch.sum(r, dim=self.axes)

        if self.normalize:
            c = torch.sum(m, dim=self.axes)
        else:
            c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

        r = torch.where(c > 0, r / c, 0.0)

        # aggregate over batch dimensions
        r = torch.mean(r)
        return r


@autojit
class WMAE(WeightedLoss):
    r"""Weighted Mean Absolute Error.

    .. math:: w𝖬𝖠𝖤(x,x̂) = \sqrt{𝔼[‖x - x̂‖_w]}
    """

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        r = predictions - targets

        m = ~torch.isnan(targets)
        r = torch.where(m, r, 0.0)
        r = self.weight * torch.abs(r)
        r = torch.sum(r, dim=self.axes)

        if self.normalize:
            c = torch.sum(m * self.weight, dim=self.axes)
        else:
            c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

        r = torch.where(c > 0, r / c, 0.0)

        # aggregate over batch dimensions
        r = torch.mean(r)
        return r


@autojit
class MSE(Loss):
    r"""Mean Square Error.

    .. math:: 𝖬𝖲𝖤(x,x̂) = 𝔼[½‖x̂-x‖^2] ∼ \tfrac{1}{2N}∑_{n=1}^N ‖x̂_n - x_n‖^2

    If the normalize option is set to True, then the normalized ℓ²-norm is used instead:

    .. math:: ‖x‖² = \frac{1}{m}∑_{i=1}^m x_i^2

    If nan_policy is set to 'omit', then NaN targets are ignored, not counting them as observations.
    In this case, the loss is computed as-if the NaN channels would not exist.

    .. math:: ‖x‖² = \frac{1}{∑ m_i} ∑_{i=1}^m [m_i ? x_i^2 : 0]

    Since it could happen that all channels are NaN, the loss is set to zero in this case.

    So, in total, there are 4 variants of the MSE loss:

    Note that this is equivalent to a weighted MSE loss with weights equal to 1.0.

    1. MSE with normalization and NaNs ignored
       .. math:: \tfrac{1}{2N}∑_{n=1}^N \tfrac{1}{∑_j m_j}∑_{i=1}^M [m_i ? (x̂_{n,i} - x_{n, i})^2 : 0]
    2. MSE with normalization and NaNs counted
       .. math:: \tfrac{1}{2N}∑_{n=1}^N \tfrac{1}{M} ∑_{i=1}^M (x̂_{n,i - x_{n,i})^2
    3. MSE without normalization and NaNs ignored
       .. math:: \tfrac{1}{2N}∑_{n=1}^N ∑_{i=1}^M [m_i ? (x̂_{n,i} - x_{n, i})^2 : 0]
    4. MSE without normalization and NaNs counted
       .. math:: \tfrac{1}{2N}∑_{n=1}^N ∑_{i=1}^M (x̂_{n,i} - x_{n, i})^2
    """

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        r = predictions - targets

        m = ~torch.isnan(targets)
        r = torch.where(m, r, 0.0)
        r = r**2
        r = torch.sum(r, dim=self.axes)

        if self.normalize:
            c = torch.sum(m, dim=self.axes)
        else:
            c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

        r = torch.where(c > 0, r / c, 0.0)

        # aggregate over batch dimensions
        r = torch.mean(r)
        return r / 2


@autojit
class WMSE(WeightedLoss):
    r"""Weighted Mean Square Error.

    .. math:: 𝗐𝖬𝖲𝖤(x,x̂) = 𝔼[‖(x - x̂)‖_w^2] ∼ \tfrac{1}{2N}∑_{n=1}^N w_i ‖x̂_n - x_n‖^2

    If the normalize option is set to True, then the normalized ℓ²-norm is used instead:

    .. math:: ‖x‖² = ∑_{i=1}^m w̃_i x_i^2 \qquad w̃_i = \frac{w_i}{∑_j w_j}

    If nan_policy is set to 'omit', then NaN targets are ignored, not counting them as observations.
    In this case, the loss is computed as-if the NaN channels would not exist. In this case,
    the existing weights are re-weighted:

    .. math:: ‖x‖² = \frac{1}{∑ m_i} ∑_{i=1}^m [m_i ?  w̃_i x_i^2 : 0]

    Since it could happen that all channels are NaN, the loss is set to zero in this case.

    So, in total, there are 4 variants of the weighted MSE loss:

    1. wMSE with normalization and NaNs ignored
       .. math:: \tfrac{1}{2N}∑_{n=1}^N \tfrac{1}{∑_j m_j w_j}∑_{i=1}^M [m_i ? w_i(x̂_{n,i} - x_{n, i})^2 : 0]
    2. wMSE with normalization and NaNs counted
       .. math:: \tfrac{1}{2N}∑_{n=1}^N \tfrac{1}{∑_j w_j}∑_{i=1}^M w_i(x̂_{n,i} - x_{n, i})^2
    3. wMSE without normalization and NaNs ignored
       .. math:: \tfrac{1}{2N}∑_{n=1}^N ∑_{i=1}^M [m_i ? w_i(x̂_{n,i} - x_{n, i})^2 : 0]
    4. wMSE without normalization and NaNs counted
       .. math:: \tfrac{1}{2N}∑_{n=1}^N ∑_{i=1}^M w_i(x̂_{n,i} - x_{n, i})^2
    """

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        r = predictions - targets

        m = ~torch.isnan(targets)
        r = torch.where(m, r, 0.0)
        r = self.weight * r**2
        r = torch.sum(r, dim=self.axes)

        if self.normalize:
            c = torch.sum(m * self.weight, dim=self.axes)
        else:
            c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

        r = torch.where(c > 0, r / c, 0.0)

        # aggregate over batch dimensions
        r = torch.mean(r)
        return r / 2


@autojit
class RMSE(Loss):
    r"""Root Mean Square Error.

    .. math:: 𝖱𝖬𝖲𝖤(x,x̂) = \sqrt{𝔼[‖x - x̂‖^2]}
    """

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        r = predictions - targets

        m = ~torch.isnan(targets)
        r = torch.where(m, r, 0.0)
        r = r**2
        r = torch.sum(r, dim=self.axes)

        if self.normalize:
            c = torch.sum(m, dim=self.axes)
        else:
            c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

        r = torch.where(c > 0, r / c, 0.0)

        # aggregate over batch dimensions
        r = torch.mean(r)
        return torch.sqrt(r / 2)


@autojit
class WRMSE(WeightedLoss):
    r"""Weighted Root Mean Square Error.

    .. math:: 𝗐𝖱𝖬𝖲𝖤(x,x̂) = \sqrt{𝔼[‖x - x̂‖_w^2]}
    """

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        r = predictions - targets

        m = ~torch.isnan(targets)
        r = torch.where(m, r, 0.0)
        r = self.weight * r**2
        r = torch.sum(r, dim=self.axes)

        if self.normalize:
            c = torch.sum(m * self.weight, dim=self.axes)
        else:
            c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

        r = torch.where(c > 0, r / c, 0.0)

        # aggregate over batch dimensions
        r = torch.mean(r)
        return torch.sqrt(r / 2)


@autojit
class TimeSeriesMSE(nn.Module):
    r"""Time-Series Mean Square Error.

    .. math:: ∑_t ∑_i \frac{[m_{t_i} ? (x̂_{t, i} - x_{t, i})^2 : 0]}{∑_t m_{t_i}}

    Or, more precisely, to avoid division by zero, we use the following

    .. math:: ∑_{ti} [m_{ti} ? (x̂_{ti} - x_{ti})^2 : 0]

    With time-normalization:

    .. math:: ∑_{ti} \frac{1}{∑_τ m_{τi}} [m_{ti} ? (x̂_{ti} - x_{ti})^2 : 0]

    with channel-normalization:

    .. math:: ∑_{ti} \frac{1}{∑_j m_{tj}} [m_{ti} ? (x̂_{ti} - x_{ti})^2 : 0]

    with both:

    .. math:: ∑_{ti} \frac{1}{∑_{τj} m_{τj}} [m_{ti} ? (x̂_{ti} - x_{ti})^2 : 0]

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
    rank: Final[int]
    r"""CONST: The number of dimensions over which the loss is computed"""

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
        self.rank = len(self.axes)

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
        return r / 2


@autojit
class TimeSeriesWMSE(WeightedLoss):
    r"""Time-Series Mean Square Error.

    .. math:: ∑_t ∑_i \frac{[m_{t_i} ? (x̂_{t, i} - x_{t, i})^2 : 0]}{∑_t m_{t_i}}

    Or, more precisely, to avoid division by zero, we use the following

    .. math:: ∑_t ∑_i [w_i>0 ? \frac{[m_{t_i} ? (x̂_{t, i} - x_{t, i})^2 : 0]}{w_i} : 0]

    where $w_i = ∑_t m_{t_i}$. Possible batch-dimensions are averaged over.

    Remark: When there are many channels,
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
        return r / 2
