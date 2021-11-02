r"""Implementations of loss functions.

Notes
-----
Contains losses in modular form.
  - See :mod:`tsdm.losses.functional` for functional implementations.
"""

from __future__ import annotations

__all__ = [
    # Classes
    "ND",
    "NRMSE",
    "Q_Quantile",
    "Q_Quantile_Loss",
]


import logging

from torch import Tensor, nn

from tsdm.losses.functional import nd, nrmse, q_quantile, q_quantile_loss

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
