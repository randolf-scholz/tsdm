r"""Loss Functions for time series.

All functions implemented in batch compatible way.
"""  # pylint: disable=R0801
import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)
__all__ = ["nd", "nrmse"]


def nd(yhat: Tensor, y: Tensor) -> Tensor:
    r"""Compute the normalized deviation score

    .. math::
        \operatorname{ND}(\hat Y, Y)
         = \frac{\sum_{t,k} |\hat Y_{t,k} -  Y_{t,k}|}{\sum_{t,k} |Y_{t,k}|}

    TODO: How to distinguish batch univariate vs single multivariate?
    => Batch makes little sense since all could have different length!

    References
    ----------

    - `Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction <https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html>`_
    - `N-BEATS: Neural basis expansion analysis for interpretable time series forecasting <https://openreview.net/forum?id=r1ecqn4YwB>`_

    Parameters
    ----------
    yhat: Tensor
    y: Tensor

    Returns
    -------
    Tensor
    """  # pylint: disable=line-too-long # noqa
    res = torch.sum(torch.abs(yhat - y), dim=(-1, -2))
    mag = torch.sum(torch.abs(y), dim=(-1, -2))
    return res / mag


def nrmse(yhat: Tensor, y: Tensor) -> Tensor:
    r"""Compute the normalized deviation score

    .. math::
        \operatorname{NRMSE}(\hat Y, Y)
         = \frac{\sqrt{ \frac{1}{T}\sum_{t,k} |\hat Y_{t,k} -  Y_{t,k}|^2 }}{\sum_{t,k} |Y_{t,k}|}

    References
    ----------

    - `Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction <https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html>`_

    Parameters
    ----------
    yhat: Tensor
    y: Tensor

    Returns
    -------
    Tensor
    """  # pylint: disable=line-too-long # noqa
    res = torch.sqrt(torch.sum(torch.abs(yhat - y) ** 2, dim=(-1, -2)))
    mag = torch.sum(torch.abs(y), dim=(-1, -2))
    return res / mag
