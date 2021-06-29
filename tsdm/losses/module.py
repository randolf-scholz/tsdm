"""
Object oriented loss functions, if you prefer.
Same name but uppercase.
"""


from torch import nn, Tensor
from .functional import nd, nrmse


class ND(nn.Module):
    r"""Compute the normalized deviation score

    .. math::
        \operatorname{ND}(\hat Y, Y)
         = \frac{\sum_{t,k} |\hat Y_{t,k} -  Y_{t,k}|}{\sum_{t,k} |Y_{t,k}|}

    TODO: How to distinguish batch univariate vs single multivariate?
    => Batch makes little sense since all could have different length!

    References
    ----------

    *  `Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction <https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html>`_
    *  `N-BEATS: Neural basis expansion analysis for interpretable time series forecasting <https://openreview.net/forum?id=r1ecqn4YwB>`_

    """  # pylint: disable=line-too-long # noqa

    @staticmethod
    def forward(yhat: Tensor, y: Tensor) -> Tensor:
        """
        Parameters
        ----------
        yhat: Tensor
        y: Tensor

        Returns
        -------
        Tensor
        """
        return nd(yhat, y)


class NRMSE(nn.Module):
    r"""Compute the normalized deviation score

    .. math::
        \operatorname{NRMSE}(\hat Y, Y)
         = \frac{\sqrt{ \frac{1}{T}\sum_{t,k} |\hat Y_{t,k} -  Y_{t,k}|^2 }}{\sum_{t,k} |Y_{t,k}|}

    References
    ----------

    *  `Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction <https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html>`_
    """  # pylint: disable=line-too-long # noqa

    @staticmethod
    def forward(yhat: Tensor, y: Tensor) -> Tensor:
        """
        Parameters
        ----------
        yhat: Tensor
        y: Tensor

        Returns
        -------
        Tensor
        """
        return nrmse(yhat, y)