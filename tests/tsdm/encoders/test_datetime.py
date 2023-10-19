r"""Test converters to masked format etc."""

import logging

from pandas import Series, date_range, testing

from tsdm.encoders import DateTimeEncoder

__logger__ = logging.getLogger(__name__)


def test_datetime_encoder() -> None:
    r"""Test whether the encoder is reversible."""
    LOGGER = __logger__.getChild(DateTimeEncoder.__name__)
    LOGGER.info("Testing.")

    # test Index
    time = date_range("2020-01-01", "2021-01-01", freq="1d")
    encoder = DateTimeEncoder()
    encoder.fit(time)
    encoded = encoder.encode(time)
    decoded = encoder.decode(encoded)
    testing.assert_index_equal(time, decoded)

    # test Series
    time = Series(time)
    encoder = DateTimeEncoder()
    encoder.fit(time)
    encoded = encoder.encode(time)
    decoded = encoder.decode(encoded)
    testing.assert_series_equal(time, decoded)
