r"""Implementations of loss functions.

Notes
-----
Contains losses in modular form.
  - See `tsdm.losses.functional` for functional implementations.
"""

__all__ = [
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
]


from typing import Final

import torch
from torch import Tensor, jit, nn

from tsdm.metrics.functional import nd, nrmse, q_quantile, q_quantile_loss
from tsdm.utils.decorators import autojit


@autojit
class ND(nn.Module):
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
    def forward(self, x: Tensor, xhat: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n), (..., n) -> ()``."""
        return nd(x, xhat)


@autojit
class NRMSE(nn.Module):
    r"""Compute the normalized root mean square error.

    .. math:: 𝖭𝖱𝖬𝖲𝖤(x, x̂) = \frac{\sqrt{ \frac{1}{T}∑_{t,k} |x̂_{t,k} - x_{t,k}|^2 }}{∑_{t,k} |x_{t,k}|}

    References
    ----------
    - | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
      | https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html
    """

    @jit.export
    def forward(self, x: Tensor, xhat: Tensor) -> Tensor:
        r"""Compute the loss value."""
        return nrmse(x, xhat)


@autojit
class Q_Quantile(nn.Module):
    r"""The q-quantile.

    .. math:: 𝖯_q(x,x̂) = \begin{cases} q |x-x̂|:& x≥x̂ \\ (1-q)|x-x̂|:& x≤x̂ \end{cases}

    References
    ----------
    - | Deep State Space Models for Time Series Forecasting
      | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """

    @jit.export
    def forward(self, x: Tensor, xhat: Tensor) -> Tensor:
        r"""Compute the loss value."""
        return q_quantile(x, xhat)


@autojit
class Q_Quantile_Loss(nn.Module):
    r"""The q-quantile loss.

    .. math:: 𝖰𝖫_q(x,x̂) = 2\frac{∑_{it}𝖯_q(x_{it},x̂_{it})}{∑_{it}|x_{it}|}

    References
    ----------
    - | Deep State Space Models for Time Series Forecasting
      | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """

    @jit.export
    def forward(self, x: Tensor, xhat: Tensor) -> Tensor:
        r"""Compute the loss value."""
        return q_quantile_loss(x, xhat)


@autojit
class MAE(nn.Module):
    r"""Mean Absolute Error.

    .. math:: 𝖬𝖠𝖤(x,x̂) = \sqrt{𝔼[‖x - x̂‖]}
    """

    ignore_nan_targets: Final[bool]
    r"""CONST: Whether to mask NaN targets, not counting them as observations."""

    def __init__(self, ignore_nan_targets: bool = True) -> None:
        super().__init__()
        self.ignore_nan_targets = ignore_nan_targets

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., m), (..., m)] → ...``."""
        r = predictions - targets

        if self.ignore_nan_targets:
            m = torch.isnan(targets)
            r = torch.where(m, 0.0, r)
            r = torch.abs(r)
            r = torch.sum(r) / torch.sum(m)
        else:
            r = torch.abs(r)
            r = torch.mean(r)

        return r


@autojit
class WMAE(nn.Module):
    r"""Weighted Mean Absolute Error.

    .. math:: w𝖬𝖠𝖤(x,x̂) = \sqrt{𝔼[‖x - x̂‖_w]}
    """

    # Constants
    rank: Final[int]
    r"""CONST: The number of dimensions of the weight tensor."""
    shape: Final[tuple[int, ...]]
    r"""CONST: The shape of the weight tensor."""
    learnable: Final[bool]
    r"""CONST: Whether the weights are learnable."""
    ignore_nan_targets: Final[bool]
    r"""CONST: Whether to ignore NaN values."""

    # Buffers
    w: Tensor
    r"""PARAM: The weight-vector."""

    def __init__(
        self,
        w: Tensor,
        /,
        normalize: bool = True,
        learnable: bool = False,
        ignore_nan_targets: bool = True,
    ):
        super().__init__()
        w = torch.tensor(w, dtype=torch.float32)
        assert torch.all(w >= 0) and torch.any(w > 0)
        w = w / torch.sum(w) if normalize else w

        self.learnable = learnable
        self.ignore_nan_targets = ignore_nan_targets
        self.w = nn.Parameter(w, requires_grad=self.learnable)
        self.rank = len(w.shape)
        self.register_buffer("FAILED", torch.tensor(float("nan")))
        self.shape = tuple(w.shape)

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., m), (..., m)] → ...``."""
        r = predictions - targets

        if self.ignore_nan_targets:
            m = torch.isnan(targets)
            r = torch.where(m, 0.0, r)
            r = self.w * torch.abs(r)
            r = torch.sum(r) / torch.sum(m)
        else:
            r = self.w * torch.abs(r)
            r = torch.mean(r)

        return r


@autojit
class MSE(nn.Module):
    r"""Mean Square Error.

    .. math:: 𝖬𝖲𝖤(x,x̂) = 𝔼[‖x - x̂‖^2]
    """

    ignore_nan_targets: Final[bool]
    r"""CONST: Whether to mask NaN targets, not counting them as observations."""

    def __init__(self, ignore_nan_targets: bool = True) -> None:
        super().__init__()
        self.ignore_nan_targets = ignore_nan_targets

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., m), (..., m)] → ...``."""
        r = predictions - targets

        if self.ignore_nan_targets:
            m = torch.isnan(targets)
            r = torch.where(m, 0.0, r)
            r = r**2
            r = torch.sum(r) / torch.sum(m)
        else:
            r = r**2
            r = torch.mean(r)

        return r


@autojit
class WMSE(nn.Module):
    r"""Weighted Mean Square Error.

    .. math:: 𝗐𝖬𝖲𝖤(x,x̂) = 𝔼[‖(x - x̂)‖_w^2]
    """

    # Constants
    rank: Final[int]
    r"""CONST: The number of dimensions of the weight tensor."""
    shape: Final[tuple[int, ...]]
    r"""CONST: The shape of the weight tensor."""
    learnable: Final[bool]
    r"""CONST: Whether the weights are learnable."""
    ignore_nan_targets: Final[bool]
    r"""CONST: Whether to ignore NaN values."""

    # Buffers
    w: Tensor
    r"""PARAM: The weight-vector."""

    def __init__(
        self,
        w: Tensor,
        /,
        normalize: bool = True,
        learnable: bool = False,
        ignore_nan_targets: bool = True,
    ):
        super().__init__()
        w = torch.tensor(w, dtype=torch.float32)
        assert torch.all(w >= 0) and torch.any(w > 0)
        w = w / torch.sum(w) if normalize else w

        self.learnable = learnable
        self.ignore_nan_targets = ignore_nan_targets
        self.w = nn.Parameter(w, requires_grad=self.learnable)
        self.rank = len(w.shape)
        self.register_buffer("FAILED", torch.tensor(float("nan")))
        self.shape = tuple(w.shape)

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., m), (..., m)] → ...``."""
        r = predictions - targets

        if self.ignore_nan_targets:
            m = torch.isnan(targets)
            r = torch.where(m, 0.0, r)
            r = self.w * r**2
            r = torch.sum(r) / torch.sum(m)
        else:
            r = self.w * r**2
            r = torch.mean(r)

        return r


@autojit
class RMSE(nn.Module):
    r"""Root Mean Square Error.

    .. math:: 𝖱𝖬𝖲𝖤(x,x̂) = \sqrt{𝔼[‖x - x̂‖^2]}
    """

    ignore_nan_targets: Final[bool]
    r"""CONST: Whether to mask NaN targets, not counting them as observations."""

    def __init__(self, ignore_nan_targets: bool = True) -> None:
        super().__init__()
        self.ignore_nan_targets = ignore_nan_targets

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., m), (..., m)] → ...``."""
        r = predictions - targets

        if self.ignore_nan_targets:
            m = torch.isnan(targets)
            r = torch.where(m, 0.0, r)
            r = r**2
            r = torch.sum(r) / torch.sum(m)
        else:
            r = r**2
            r = torch.mean(r)

        return torch.sqrt(r)


@autojit
class WRMSE(nn.Module):
    r"""Weighted Root Mean Square Error.

    .. math:: 𝗐𝖱𝖬𝖲𝖤(x,x̂) = \sqrt{𝔼[‖x - x̂‖_w^2]}
    """

    # Constants
    rank: Final[int]
    r"""CONST: The number of dimensions of the weight tensor."""
    shape: Final[tuple[int, ...]]
    r"""CONST: The shape of the weight tensor."""
    learnable: Final[bool]
    r"""CONST: Whether the weights are learnable."""
    ignore_nan_targets: Final[bool]
    r"""CONST: Whether to ignore NaN values."""

    # Buffers
    w: Tensor
    r"""PARAM: The weight-vector."""

    def __init__(
        self,
        w: Tensor,
        /,
        normalize: bool = True,
        learnable: bool = False,
        ignore_nan_targets: bool = True,
    ):
        super().__init__()
        w = torch.tensor(w, dtype=torch.float32)
        assert torch.all(w >= 0) and torch.any(w > 0)
        w = w / torch.sum(w) if normalize else w

        self.learnable = learnable
        self.ignore_nan_targets = ignore_nan_targets
        self.w = nn.Parameter(w, requires_grad=self.learnable)
        self.rank = len(w.shape)
        self.register_buffer("FAILED", torch.tensor(float("nan")))
        self.shape = tuple(w.shape)

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., m), (..., m)] → ...``."""
        r = predictions - targets

        if self.ignore_nan_targets:
            m = torch.isnan(targets)
            r = torch.where(m, 0.0, r)
            r = self.w * r**2
            r = torch.sum(r) / torch.sum(m)
        else:
            r = self.w * r**2
            r = torch.mean(r)

        return torch.sqrt(r)
