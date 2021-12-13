r"""Implementations of loss functions.

Notes
-----
Contains losses in functional form.
  - See :mod:`tsdm.losses.modular` for modular implementations.
"""

__all__ = [  # Functions
    "nd",
    "nrmse",
    "rmse",
    "q_quantile",
    "q_quantile_loss",
]

import logging

import torch
from torch import Tensor, jit

__logger__ = logging.getLogger(__name__)


@jit.script
def nd(x: Tensor, xhat: Tensor, eps: float = 2 ** -24) -> Tensor:
    r"""Compute the normalized deviation score.

    .. math::
        𝖭𝖣(x, x̂) = \frac{∑_{t,k} |x̂_{t,k} -  x_{t,k}|}{∑_{t,k} |x_{t,k}|}

    TODO: How to distinguish batch univariate vs single multivariate?
    => Batch makes little sense since all could have different length!

    References
    ----------
    - `Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction <https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html>`_
    - `N-BEATS: Neural basis expansion analysis for interpretable time series forecasting <https://openreview.net/forum?id=r1ecqn4YwB>`_

    Parameters
    ----------
    xhat: Tensor
    x: Tensor

    Returns
    -------
    Tensor
    """  # pylint: disable=line-too-long # noqa
    res = torch.sum(torch.abs(xhat - x), dim=(-1, -2))
    mag = torch.sum(torch.abs(x), dim=(-1, -2))
    mag = torch.maximum(mag, torch.tensor(eps, dtype=x.dtype, device=x.device))
    return torch.mean(res / mag)  # get rid of any batch dimensions


@jit.script
def nrmse(x: Tensor, xhat: Tensor, eps: float = 2 ** -24) -> Tensor:
    r"""Compute the normalized deviation score.

    .. math::
        𝖭𝖱𝖬𝖲𝖤(x, x̂) = \frac{\sqrt{ \frac{1}{T}∑_{t,k} |x̂_{t,k} - x_{t,k}|^2 }}{∑_{t,k} |x_{t,k}|}

    References
    ----------
    - `Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
      <https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html>`_

    Parameters
    ----------
    xhat: Tensor
    x: Tensor

    Returns
    -------
    Tensor
    """
    res = torch.sqrt(torch.sum(torch.abs(xhat - x) ** 2, dim=(-1, -2)))
    mag = torch.sum(torch.abs(x), dim=(-1, -2))
    mag = torch.maximum(mag, torch.tensor(eps, dtype=x.dtype, device=x.device))
    return torch.mean(res / mag)  # get rid of any batch dimensions


@jit.script
def q_quantile(x: Tensor, xhat: Tensor, q: float = 0.5) -> Tensor:
    r"""Return the q-quantile.

    .. math::
         𝖯_q(x,x̂) = \begin{cases} q |x-x̂|:& x≥x̂ \\ (1-q)|x-x̂|:& x≤x̂ \end{cases}

     References
     ----------
     - `Deep State Space Models for Time Series Forecasting <https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html>`_

     Parameters
     ----------
     x: Tensor
     xhat: Tensor
     q: float

     Returns
     -------
     Tensor
    """  # pylint: disable=line-too-long # noqa
    residual = x - xhat
    return torch.max((q - 1) * residual, q * residual)


@jit.script
def q_quantile_loss(x: Tensor, xhat: Tensor, q: float = 0.5) -> Tensor:
    r"""Return the q-quantile loss.

    .. math::
        𝖰𝖫_q(x,x̂) = 2\frac{∑_{it}𝖯_q(x_{it},x̂_{it})}{∑_{it}|x_{it}|}

    References
    ----------
    - `Deep State Space Models for Time Series Forecasting <https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html>`_

    Parameters
    ----------
    x: Tensor
    xhat: Tensor
    q: float

    Returns
    -------
    Tensor
    """  # pylint: disable=line-too-long # noqa
    return 2 * torch.sum(q_quantile(x, xhat, q)) / torch.sum(torch.abs(x))


@jit.script
def rmse(
    x: Tensor,
    xhat: Tensor,
) -> Tensor:
    r"""Compute the RMSE.

    .. math::
        𝗋𝗆𝗌𝖾(x,x̂) = \sqrt{𝔼[|x - x̂|^2]}

    Parameters
    ----------
    x: Tensor,
    xhat: Tensor,

    Returns
    -------
    Tensor
    """
    return torch.sqrt(torch.mean((x - xhat) ** 2))
