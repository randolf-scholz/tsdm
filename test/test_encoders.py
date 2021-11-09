r"""Test converters to masked format etc."""

import logging

from pandas import Series, date_range, testing

from tsdm.encoders.modular import DateTimeEncoder

LOGGER = logging.getLogger(__name__)


def test_datetime_encoder():
    r"""Test whether the encoder is reversible."""
    time = date_range("2020-01-01", "2021-01-01", freq="1d")
    encoder = DateTimeEncoder()
    encoder.fit(time)
    encoded = encoder.encode(time)
    decoded = encoder.decode(encoded)
    testing.assert_series_equal(time, decoded)

    time = Series(time)
    encoder = DateTimeEncoder()
    encoder.fit(time)
    encoded = encoder.encode(time)
    decoded = encoder.decode(encoded)
    testing.assert_series_equal(time, decoded)


def __main__():
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Testing MASKED_FORMAT started!")
    test_datetime_encoder()
    LOGGER.info("Testing MASKED_FORMAT finished!")


if __name__ == "__main__":
    __main__()
