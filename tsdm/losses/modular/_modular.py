r"""Implementations of loss functions.

Notes
-----
Contains losses in modular form.
  - See :mod:`tsdm.losses.functional` for functional implementations.
"""

__all__ = [
    # Classes
    "ND",
    "NRMSE",
    "Q_Quantile",
    "Q_Quantile_Loss",
]


import logging
from typing import Final

import torch
from torch import Tensor, nn

from tsdm.losses.functional import nd, nrmse, q_quantile, q_quantile_loss, rmse

__logger__ = logging.getLogger(__name__)


class ND(nn.Module):
    r"""Compute the normalized deviation score.

    .. math::
        𝖭𝖣(x, x̂) = \frac{∑_{t,k} |x̂_{t,k} -  x_{t,k}|}{∑_{t,k} |x_{t,k}|}

    TODO: How to distinguish batch univariate vs single multivariate?
    => Batch makes little sense since all could have different length!

    References
    ----------
    -  `Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction <https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html>`_
    -  `N-BEATS: Neural basis expansion analysis for interpretable time series forecasting <https://openreview.net/forum?id=r1ecqn4YwB>`_
    """  # pylint: disable=line-too-long # noqa

    @staticmethod
    def forward(x: Tensor, xhat: Tensor) -> Tensor:
        r"""Compute the loss value.

        Parameters
        ----------
        x: Tensor
        xhat: Tensor

        Returns
        -------
        Tensor
        """
        return nd(x, xhat)


class NRMSE(nn.Module):
    r"""Compute the normalized deviation score.

    .. math::
        𝖭𝖱𝖬𝖲𝖤(x, x̂) = \frac{\sqrt{ \frac{1}{T}∑_{t,k} |x̂_{t,k} - x_{t,k}|^2 }}{∑_{t,k} |x_{t,k}|}

    References
    ----------
    -  `Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction <https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html>`_
    """  # pylint: disable=line-too-long # noqa

    @staticmethod
    def forward(x: Tensor, xhat: Tensor) -> Tensor:
        r"""Compute the loss value.

        Parameters
        ----------
        x: Tensor
        xhat: Tensor

        Returns
        -------
        Tensor
        """
        return nrmse(x, xhat)


class Q_Quantile(nn.Module):
    r"""The q-quantile.

    .. math::
        𝖯_q(x,x̂) = \begin{cases} q |x-x̂|:& x≥x̂ \\ (1-q)|x-x̂|:& x≤x̂ \end{cases}

    References
    ----------
    - `Deep State Space Models for Time Series Forecasting <https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html>`_
    """  # pylint: disable=line-too-long # noqa

    @staticmethod
    def forward(x: Tensor, xhat: Tensor) -> Tensor:
        r"""Compute the loss value.

        Parameters
        ----------
        x: Tensor
        xhat: Tensor

        Returns
        -------
        Tensor
        """
        return q_quantile(x, xhat)


class Q_Quantile_Loss(nn.Module):
    r"""The q-quantile loss.

    .. math::
        𝖰𝖫_q(x,x̂) = 2\frac{∑_{it}𝖯_q(x_{it},x̂_{it})}{∑_{it}|x_{it}|}

    References
    ----------
    - `Deep State Space Models for Time Series Forecasting <https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html>`_
    """  # pylint: disable=line-too-long # noqa

    @staticmethod
    def forward(x: Tensor, xhat: Tensor) -> Tensor:
        r"""Compute the loss value.

        Parameters
        ----------
        x: Tensor
        xhat: Tensor

        Returns
        -------
        Tensor
        """
        return q_quantile_loss(x, xhat)


class WRMSE(nn.Module):
    r"""Weighted Root Mean Square Error.

    .. math::

        (1/m)∑_m (1/n)∑_n w(x_{n,m}- x_{n,m})^2
    """

    # Constants
    rank: Final[int]
    r"""CONST: The number of dimensions of the weight tensor."""
    shape: Final[list[int]]
    r"""CONST: The shape of the weight tensor."""
    channel_wise: Final[bool]
    r"""CONST: Whether to compute the it channel wise."""
    ignore_nan: Final[bool]
    r"""CONST: Whether to ignore NaN-values."""
    # Buffers
    w: Tensor
    r"""BUFFER: The weight-vector."""

    def __init__(
        self,
        w: Tensor,
        ignore_nan: bool = True,
        channel_wise: bool = True,
        normalize: bool = True,
    ):
        r"""Compute the weighted RMSE.

        Channel-wise: RMSE(RMSE(channel))
        Non-channel-wise: RMSE(flatten(results))

        Parameters
        ----------
        w: Tensor
        ignore_nan: bool = True
        channel_wise: bool = True
            If true, compute mean across channels first and then accumulate channels
        normalize: bool = True
        """
        super().__init__()
        assert torch.all(w >= 0) and torch.any(w > 0)
        self.w = w / torch.sum(w) if normalize else w
        self.rank = len(w.shape)
        self.ignore_nan = ignore_nan
        self.channel_wise = channel_wise
        self.register_buffer("FAILED", torch.tensor(float("nan")))
        self.shape = list(w.shape)

    def forward(self, x: Tensor, xhat: Tensor) -> Tensor:
        r"""Signature: ``...𝐦, ...𝐦 → 0``.

        Parameters
        ----------
        x: Tensor
        xhat: Tensor

        Returns
        -------
        Tensor
        """
        assert x.shape[-self.rank :] == self.shape
        batch_dims = list(range(len(x.shape) - self.rank))

        mask = torch.isnan(x)
        # the residuals, shape: ...𝐦
        r = self.w * (x - xhat) ** 2

        # xhat is not allowed to be nan if x isn't.
        if torch.any(torch.isnan(xhat) & ~mask):
            raise RuntimeError("Observations have NaN entries when Targets do not!")

        if self.ignore_nan:
            if self.channel_wise:
                return rmse(torch.sqrt(torch.nanmean(r, dim=batch_dims)))
            return torch.sqrt(torch.nanmean(r, dim=batch_dims))

        if self.channel_wise:
            rmse(torch.sqrt(torch.mean(r, dim=batch_dims)))
        return torch.sqrt(torch.mean(r))
