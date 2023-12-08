r"""Test converters to masked format etc."""

import logging

from pandas import NA, DataFrame, testing

from tsdm.encoders.transforms import make_masked_format

__logger__ = logging.getLogger(__name__)


def test_make_masked_format() -> None:
    r"""Using example taken from Figure 2 in [1].

    References
    ----------
    1. Recurrent Neural Networks for Multivariate Time Series with Missing Values
       Che et al., Nature 2017
    """
    LOGGER = __logger__.getChild(make_masked_format.__name__)
    LOGGER.info("Testing.")

    x_raw = [[47, 49, NA, 40, NA, 43, 55], [NA, 15, 14, NA, NA, NA, 15]]
    t_raw = [0, 0.1, 0.6, 1.6, 2.2, 2.5, 3.1]
    d_raw = [[0.0, 0.1, 0.5, 1.5, 0.6, 0.9, 0.6], [0.0, 0.1, 0.5, 1.0, 1.6, 1.9, 2.5]]
    m_raw = [[1, 1, 0, 1, 0, 1, 1], [0, 1, 1, 0, 0, 0, 1]]

    data = DataFrame(x_raw, columns=t_raw).T
    mask = DataFrame(m_raw, columns=t_raw).T
    diff = DataFrame(d_raw, columns=t_raw).T
    x, m, d = make_masked_format(data)

    testing.assert_frame_equal(x, data, check_dtype=False)
    testing.assert_frame_equal(m, mask, check_dtype=False)
    testing.assert_frame_equal(d, diff, check_dtype=False)
